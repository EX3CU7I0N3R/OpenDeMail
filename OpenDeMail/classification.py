import argparse
import json
import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from OpenDeMail.modules.mailDB import MailDB


logging.basicConfig(
    filename="opendemail.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

PROMO_PATTERN = re.compile(
    r"\b(verify|urgent|winner|offer|save|free|discount|limited|sale|special|alert|deal|bonus|reward|cashback)\b",
    re.IGNORECASE,
)
PHISHING_PATTERN = re.compile(
    r"\b(password|account|login|click|confirm|suspend|reset|otp|security|bank|payment|invoice|gift)\b",
    re.IGNORECASE,
)
TRUSTED_TRANSACTIONAL_PATTERN = re.compile(
    r"\b(statement|receipt|invoice|notification|alert|update|confirmation|application|document|internship)\b",
    re.IGNORECASE,
)
FREE_MAILBOX_DOMAINS = {"gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "live.com"}
LABEL_ORDER = ["likely_ham", "suspicious", "likely_spam"]
REVIEW_EXPORT_COLUMNS = [
    "email_id",
    "message_id",
    "sender_domain",
    "sender_email",
    "subject",
    "predicted_label",
    "spam_score",
    "ground_truth_label",
    "review_notes",
    "reviewed_at",
]


class EmailClassifier:
    def __init__(
        self,
        db_path: str = "emails.db",
        output_dir: str = "classification_output",
        review_csv_path: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        incremental: bool = False,
    ) -> None:
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.review_csv_path = Path(review_csv_path) if review_csv_path else None
        self.start_date = pd.to_datetime(start_date).normalize() if start_date else None
        self.end_date = pd.to_datetime(end_date).normalize() if end_date else None
        self.incremental = incremental
        self.output_dir.mkdir(exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        db = MailDB(self.db_path)
        try:
            query = """
                SELECT
                    id,
                    message_id,
                    recipient_email,
                    sender_domain,
                    sender_email,
                    subject,
                    date_header,
                    mailer,
                    spf_result,
                    dkim_result,
                    dmarc_result,
                    received,
                    x_received,
                    content_type,
                    processed_category,
                    processed_flag,
                    spam_score,
                    spam_label,
                    spam_reasons,
                    raw_headers_json,
                    created_at
                FROM emails
            """
            frame = pd.read_sql_query(query, db.conn)
        finally:
            db.close()

        if frame.empty:
            logging.info("Loaded 0 emails for classification")
            return frame

        frame["parsed_date"] = pd.to_datetime(frame["date_header"], errors="coerce", utc=True)

        if self.start_date is not None:
            frame = frame[frame["parsed_date"].dt.tz_localize(None) >= self.start_date]
        if self.end_date is not None:
            inclusive_end = self.end_date + pd.Timedelta(days=1)
            frame = frame[frame["parsed_date"].dt.tz_localize(None) < inclusive_end]
        if self.incremental:
            frame = frame[
                frame["processed_category"].fillna("").eq("")
                | frame["spam_label"].fillna("").eq("")
                | frame["processed_flag"].fillna("").eq("")
            ]

        frame = frame.reset_index(drop=True)
        logging.info("Loaded %s emails for classification after filters", len(frame))
        return frame

    @staticmethod
    def _safe_text(value: object) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _extract_top_terms(subjects: pd.Series) -> list[str]:
        tokens = [
            token
            for token in re.findall(r"[a-z]{4,}", " ".join(subjects.astype(str).tolist()).lower())
            if token not in {"from", "with", "your", "this", "have", "that", "today", "email"}
        ]
        if not tokens:
            return []
        return pd.Series(tokens).value_counts().head(5).index.tolist()

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        if denominator == 0:
            return 0.0
        return round(float(numerator) / float(denominator), 4)

    @staticmethod
    def _count_matching_terms(text: str, pattern: re.Pattern[str]) -> int:
        return len(pattern.findall(text or ""))

    def engineer_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        working = frame.copy()
        for column in [
            "recipient_email",
            "sender_domain",
            "sender_email",
            "subject",
            "mailer",
            "content_type",
            "spf_result",
            "dkim_result",
            "dmarc_result",
            "received",
            "x_received",
        ]:
            working[column] = working[column].fillna("")

        working["recipient_email"] = working["recipient_email"].str.lower()
        working["sender_domain"] = working["sender_domain"].str.lower()
        working["sender_email"] = working["sender_email"].str.lower()
        working["spf_result"] = working["spf_result"].str.lower()
        working["dkim_result"] = working["dkim_result"].str.lower()
        working["dmarc_result"] = working["dmarc_result"].str.lower()

        working["subject_token_count"] = working["subject"].str.split().str.len().fillna(0)
        working["received_hops"] = working["received"].str.count("\n").fillna(0) + 1
        working.loc[working["received"].eq(""), "received_hops"] = 0
        working["x_received_hops"] = working["x_received"].str.count("\n").fillna(0) + 1
        working.loc[working["x_received"].eq(""), "x_received_hops"] = 0
        working["spf_pass"] = working["spf_result"].eq("pass").astype(int)
        working["dkim_pass"] = working["dkim_result"].eq("pass").astype(int)
        working["dmarc_pass"] = working["dmarc_result"].eq("pass").astype(int)
        working["auth_pass_count"] = working[["spf_pass", "dkim_pass", "dmarc_pass"]].sum(axis=1)
        working["auth_failure_count"] = 3 - working["auth_pass_count"]
        working["promo_term_hits"] = working["subject"].str.count(PROMO_PATTERN).fillna(0)
        working["phishing_term_hits"] = working["subject"].str.count(PHISHING_PATTERN).fillna(0)
        working["transactional_term_hits"] = working["subject"].map(
            lambda text: self._count_matching_terms(str(text), TRUSTED_TRANSACTIONAL_PATTERN)
        )
        working["exclamation_count"] = working["subject"].str.count("!").fillna(0)
        working["uppercase_ratio"] = working["subject"].map(
            lambda text: (
                sum(1 for char in str(text) if char.isupper()) / max(sum(1 for char in str(text) if char.isalpha()), 1)
            )
        )
        working["is_html"] = working["content_type"].str.contains("html", case=False, regex=False).astype(int)
        working["free_mailbox_sender"] = working["sender_domain"].isin(FREE_MAILBOX_DOMAINS).astype(int)
        working["unknown_sender_domain"] = working["sender_domain"].eq("").astype(int)

        working["domain_frequency"] = working.groupby("sender_domain")["id"].transform("count")
        working["sender_frequency"] = working.groupby("sender_email")["id"].transform("count")
        working["sender_recipient_frequency"] = (
            working.groupby(["sender_email", "recipient_email"])["id"].transform("count")
        )
        working["recipient_count_for_sender"] = working.groupby("sender_email")["recipient_email"].transform("nunique")

        domain_auth_ratio = working.groupby("sender_domain")["auth_pass_count"].transform("mean") / 3.0
        sender_auth_ratio = working.groupby("sender_email")["auth_pass_count"].transform("mean") / 3.0

        working["prior_interaction"] = (
            working["sender_email"].ne("") & working["sender_frequency"].gt(1)
        ).astype(int)
        working["stable_sender_recipient_pattern"] = (
            working["sender_email"].ne("")
            & working["recipient_email"].ne("")
            & working["sender_recipient_frequency"].ge(2)
        ).astype(int)
        working["recipient_specific_sender"] = (
            working["sender_email"].ne("") & working["recipient_count_for_sender"].le(2)
        ).astype(int)
        working["recurring_transactional_domain"] = (
            working["sender_domain"].ne("")
            & working["domain_frequency"].ge(3)
            & domain_auth_ratio.ge(0.85)
            & working["transactional_term_hits"].ge(1)
        ).astype(int)
        working["trusted_sender_profile"] = (
            (
                working["sender_email"].ne("")
                & working["sender_frequency"].ge(3)
                & sender_auth_ratio.ge(0.66)
                & working["stable_sender_recipient_pattern"].eq(1)
            )
            | (
                working["sender_domain"].ne("")
                & working["domain_frequency"].ge(5)
                & domain_auth_ratio.ge(0.90)
                & working["transactional_term_hits"].ge(1)
            )
        ).astype(int)
        working["trusted_free_mailbox_sender"] = (
            working["free_mailbox_sender"].eq(1)
            & working["stable_sender_recipient_pattern"].eq(1)
            & working["prior_interaction"].eq(1)
        ).astype(int)

        working["combined_text"] = (
            working["subject"].map(self._safe_text)
            + " domain:"
            + working["sender_domain"].map(self._safe_text)
            + " mailer:"
            + working["mailer"].map(self._safe_text)
            + " content:"
            + working["content_type"].map(self._safe_text)
            + " spf:"
            + working["spf_result"].map(self._safe_text)
            + " dkim:"
            + working["dkim_result"].map(self._safe_text)
            + " dmarc:"
            + working["dmarc_result"].map(self._safe_text)
        )
        working.loc[working["combined_text"].str.strip().eq(""), "combined_text"] = "unknown email metadata"
        return working

    def build_feature_matrix(self, working: pd.DataFrame):
        vectorizer = TfidfVectorizer(stop_words="english", max_features=600, ngram_range=(1, 2))
        text_features = vectorizer.fit_transform(working["combined_text"])

        numeric_features = csr_matrix(
            working[
                [
                    "subject_token_count",
                    "received_hops",
                    "x_received_hops",
                    "spf_pass",
                    "dkim_pass",
                    "dmarc_pass",
                    "promo_term_hits",
                    "phishing_term_hits",
                    "transactional_term_hits",
                    "exclamation_count",
                    "uppercase_ratio",
                    "is_html",
                    "free_mailbox_sender",
                    "unknown_sender_domain",
                    "domain_frequency",
                    "sender_frequency",
                    "sender_recipient_frequency",
                    "prior_interaction",
                    "stable_sender_recipient_pattern",
                    "recurring_transactional_domain",
                    "trusted_sender_profile",
                    "trusted_free_mailbox_sender",
                ]
            ].to_numpy(dtype=float)
        )
        feature_matrix = hstack([text_features, numeric_features]).tocsr()
        return feature_matrix, vectorizer

    @staticmethod
    def _select_cluster_count(feature_matrix) -> int:
        sample_size = feature_matrix.shape[0]
        if sample_size <= 2:
            return 1
        if sample_size < 12:
            return 2

        upper_bound = min(8, sample_size - 1)
        best_k = 4
        best_score = -1.0
        for cluster_count in range(3, upper_bound + 1):
            model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
            labels = model.fit_predict(feature_matrix)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(feature_matrix, labels, sample_size=min(1000, sample_size), random_state=42)
            if score > best_score:
                best_score = score
                best_k = cluster_count
        return best_k

    @staticmethod
    def _summarize_cluster_name(cluster_frame: pd.DataFrame) -> str:
        top_domain = cluster_frame["sender_domain"].replace("", "unknown").mode().iloc[0]
        top_terms = EmailClassifier._extract_top_terms(cluster_frame["subject"])
        keyword = top_terms[0] if top_terms else "general"
        pass_rate = cluster_frame[["spf_pass", "dkim_pass", "dmarc_pass"]].mean().mean()
        promo_rate = cluster_frame["promo_term_hits"].gt(0).mean()
        phishing_rate = cluster_frame["phishing_term_hits"].gt(0).mean()
        trust_rate = cluster_frame["trusted_sender_profile"].mean()

        if trust_rate >= 0.55 and pass_rate >= 0.70:
            prefix = "trusted"
        elif pass_rate < 0.45 or phishing_rate > 0.30:
            prefix = "unverified"
        elif promo_rate > 0.40:
            prefix = "marketing"
        else:
            prefix = "transactional"
        return f"{prefix}_{top_domain}_{keyword}"

    def assign_clusters(self, working: pd.DataFrame, feature_matrix, vectorizer):
        if len(working) == 1:
            working = working.copy()
            working["cluster_id"] = 0
            working["processed_category"] = self._summarize_cluster_name(working)
            top_terms_per_cluster = {0: self._extract_top_terms(working["subject"])}
            return working, {0: working["processed_category"].iloc[0]}, top_terms_per_cluster

        cluster_count = self._select_cluster_count(feature_matrix)
        model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
        labels = model.fit_predict(feature_matrix)

        working = working.copy()
        working["cluster_id"] = labels
        cluster_name_map = {}
        for cluster_id in sorted(working["cluster_id"].unique()):
            cluster_name_map[cluster_id] = self._summarize_cluster_name(working[working["cluster_id"] == cluster_id])
        working["processed_category"] = working["cluster_id"].map(cluster_name_map)

        feature_names = np.array(vectorizer.get_feature_names_out())
        centroids = model.cluster_centers_[:, : len(feature_names)]
        top_terms_per_cluster = {}
        for cluster_id, centroid in enumerate(centroids):
            top_terms_per_cluster[int(cluster_id)] = feature_names[np.argsort(centroid)[-8:][::-1]].tolist()
        return working, cluster_name_map, top_terms_per_cluster

    @staticmethod
    def _score_single_email(row: pd.Series) -> tuple[float, str, str]:
        score = 0.0
        reasons: list[str] = []
        trusted_context = bool(row.get("trusted_sender_profile", 0))
        trusted_free_mailbox = bool(row.get("trusted_free_mailbox_sender", 0))
        recurring_transactional = bool(row.get("recurring_transactional_domain", 0))
        auth_failures = int(row.get("auth_failure_count", 0))

        if not row["spf_pass"]:
            penalty = 0.8 if trusted_context else 1.4
            score += penalty
            reasons.append("SPF did not pass")
        if not row["dkim_pass"]:
            penalty = 0.8 if trusted_context else 1.4
            score += penalty
            reasons.append("DKIM did not pass")
        if not row["dmarc_pass"]:
            penalty = 0.7 if trusted_context else 1.2
            score += penalty
            reasons.append("DMARC did not pass")

        if row["promo_term_hits"] > 0:
            promo_penalty = 0.35 if recurring_transactional else 0.6
            score += min(1.8, promo_penalty * float(row["promo_term_hits"]))
            reasons.append(f"Promotional language detected ({int(row['promo_term_hits'])} hit(s))")

        if row["phishing_term_hits"] > 0:
            if trusted_context and auth_failures <= 1:
                phishing_penalty = 0.2
                reasons.append(
                    f"Sensitive keywords detected in a trusted sender context ({int(row['phishing_term_hits'])} hit(s))"
                )
            elif recurring_transactional and auth_failures <= 1:
                phishing_penalty = 0.3
                reasons.append(
                    f"Sensitive keywords detected in a recurring transactional context ({int(row['phishing_term_hits'])} hit(s))"
                )
            else:
                phishing_penalty = 0.7
                reasons.append(f"Security/payment keywords detected ({int(row['phishing_term_hits'])} hit(s))")
            score += min(2.1, phishing_penalty * float(row["phishing_term_hits"]))

        if row["received_hops"] >= 5:
            score += 0.7
            reasons.append("High number of mail transfer hops")
        elif row["received_hops"] == 4:
            score += 0.4
            reasons.append("Moderately high number of mail transfer hops")

        if row["exclamation_count"] >= 2:
            score += 0.5
            reasons.append("Aggressive punctuation in subject")
        if row["uppercase_ratio"] >= 0.45 and row["subject_token_count"] >= 3:
            score += 0.5
            reasons.append("High uppercase ratio in subject")
        if row["free_mailbox_sender"] and row["promo_term_hits"] > 0 and not trusted_free_mailbox:
            score += 0.6
            reasons.append("Free mailbox domain used with promotional subject")
        if row["unknown_sender_domain"]:
            score += 0.7
            reasons.append("Sender domain missing")
        if row["is_html"] and row["promo_term_hits"] > 0 and not recurring_transactional:
            score += 0.4
            reasons.append("HTML email combined with promotional content")

        category = str(row["processed_category"])
        if category.startswith("unverified_") and not trusted_context:
            score += 0.8
            reasons.append("Cluster behaves like an unverified sender group")
        elif category.startswith("marketing_") and not recurring_transactional:
            score += 0.4
            reasons.append("Cluster behaves like a marketing sender group")

        trust_credits = 0.0
        trust_reasons = []
        if trusted_context:
            trust_credits += 1.0
            trust_reasons.append("Recurring sender-recipient history lowers risk")
        if recurring_transactional:
            trust_credits += 0.8
            trust_reasons.append("Recurring authenticated transactional domain lowers risk")
        if row.get("prior_interaction", 0):
            trust_credits += 0.3
            trust_reasons.append("Repeated sender history lowers risk")
        if row.get("recipient_specific_sender", 0):
            trust_credits += 0.2
            trust_reasons.append("Stable sender-recipient routing lowers risk")
        if trusted_free_mailbox:
            trust_credits += 0.6
            trust_reasons.append("Trusted personal mailbox history lowers risk")

        if trust_credits:
            score = max(0.0, score - trust_credits)
            reasons.extend(trust_reasons)

        normalized_score = round(min(100.0, (score / 8.5) * 100.0), 2)
        if normalized_score >= 70:
            label = "likely_spam"
        elif normalized_score >= 40:
            label = "suspicious"
        else:
            label = "likely_ham"

        if not reasons:
            reasons.append("No strong spam indicators were triggered")
        return normalized_score, label, "; ".join(dict.fromkeys(reasons))

    def score_emails(self, clustered_frame: pd.DataFrame) -> pd.DataFrame:
        scored = clustered_frame.copy()
        spam_results = scored.apply(self._score_single_email, axis=1, result_type="expand")
        spam_results.columns = ["spam_score", "spam_label", "spam_reasons"]
        scored[["spam_score", "spam_label", "spam_reasons"]] = spam_results
        scored["processed_flag"] = np.where(
            scored["spam_label"].isin(["likely_spam", "suspicious"]),
            "review",
            "normal",
        )
        return scored

    @staticmethod
    def _project_clusters(feature_matrix, row_count: int) -> tuple[np.ndarray, np.ndarray]:
        if row_count == 1:
            return np.array([0.0]), np.array([0.0])
        svd = TruncatedSVD(n_components=2, random_state=42)
        projection = svd.fit_transform(feature_matrix)
        return projection[:, 0], projection[:, 1]

    def persist_results(self, classified_frame: pd.DataFrame) -> None:
        db = MailDB(self.db_path)
        try:
            updates = classified_frame[
                ["id", "processed_category", "processed_flag", "spam_score", "spam_label", "spam_reasons"]
            ].itertuples(index=False, name=None)
            db.bulk_update_classification(updates)
        finally:
            db.close()

    def import_review_csv(self) -> int:
        if self.review_csv_path is None or not self.review_csv_path.exists():
            return 0

        reviews = pd.read_csv(self.review_csv_path)
        required_columns = {"email_id", "ground_truth_label"}
        if not required_columns.issubset(reviews.columns):
            raise ValueError(
                f"Review CSV must contain columns: {', '.join(sorted(required_columns))}"
            )

        reviews = reviews.copy()
        reviews["ground_truth_label"] = reviews["ground_truth_label"].fillna("").str.strip()
        reviews = reviews[reviews["ground_truth_label"].isin(LABEL_ORDER)]
        if reviews.empty:
            return 0

        if "review_notes" not in reviews.columns:
            reviews["review_notes"] = ""
        else:
            reviews["review_notes"] = reviews["review_notes"].fillna("")
        if "reviewed_at" not in reviews.columns:
            reviews["reviewed_at"] = ""
        else:
            reviews["reviewed_at"] = reviews["reviewed_at"].fillna("")

        db = MailDB(self.db_path)
        try:
            db.bulk_upsert_manual_reviews(
                reviews[["email_id", "ground_truth_label", "review_notes", "reviewed_at"]].itertuples(
                    index=False,
                    name=None,
                )
            )
        finally:
            db.close()
        return int(len(reviews))

    def load_review_labels(self) -> pd.DataFrame:
        db = MailDB(self.db_path)
        try:
            reviews = pd.read_sql_query(
                """
                SELECT email_id, ground_truth_label, review_notes, reviewed_at
                FROM review_labels
                """,
                db.conn,
            )
        finally:
            db.close()
        return reviews

    def evaluate_predictions(self, classified_frame: pd.DataFrame) -> dict:
        reviews = self.load_review_labels()
        empty_summary = {
            "reviewed_count": 0,
            "exact_match_accuracy": None,
            "review_precision": None,
            "review_recall": None,
            "likely_spam_precision": None,
            "likely_spam_recall": None,
            "confusion_matrix": [],
            "top_false_positives": [],
            "top_false_negatives": [],
        }
        if reviews.empty:
            return empty_summary

        evaluated = classified_frame.merge(reviews, how="inner", left_on="id", right_on="email_id")
        if evaluated.empty:
            return empty_summary

        evaluated["ground_truth_label"] = pd.Categorical(evaluated["ground_truth_label"], categories=LABEL_ORDER, ordered=True)
        evaluated["predicted_label"] = pd.Categorical(evaluated["spam_label"], categories=LABEL_ORDER, ordered=True)

        review_predicted = evaluated["spam_label"].isin(["suspicious", "likely_spam"])
        review_actual = evaluated["ground_truth_label"].isin(["suspicious", "likely_spam"])
        spam_predicted = evaluated["spam_label"].eq("likely_spam")
        spam_actual = evaluated["ground_truth_label"].eq("likely_spam")

        confusion = []
        confusion_table = pd.crosstab(evaluated["ground_truth_label"], evaluated["spam_label"], dropna=False)
        for actual_label in LABEL_ORDER:
            for predicted_label in LABEL_ORDER:
                confusion.append(
                    {
                        "actual_label": actual_label,
                        "predicted_label": predicted_label,
                        "count": int(confusion_table.get(predicted_label, pd.Series(dtype=int)).get(actual_label, 0)),
                    }
                )

        false_positives = evaluated[(review_predicted) & (evaluated["ground_truth_label"] == "likely_ham")]
        false_negatives = evaluated[(~review_predicted) & (review_actual)]

        return {
            "reviewed_count": int(len(evaluated)),
            "ground_truth_counts": evaluated["ground_truth_label"].value_counts().to_dict(),
            "predicted_counts": evaluated["spam_label"].value_counts().to_dict(),
            "exact_match_accuracy": round(float((evaluated["ground_truth_label"] == evaluated["predicted_label"]).mean()), 4),
            "review_precision": self._safe_ratio((review_predicted & review_actual).sum(), review_predicted.sum()),
            "review_recall": self._safe_ratio((review_predicted & review_actual).sum(), review_actual.sum()),
            "likely_spam_precision": self._safe_ratio((spam_predicted & spam_actual).sum(), spam_predicted.sum()),
            "likely_spam_recall": self._safe_ratio((spam_predicted & spam_actual).sum(), spam_actual.sum()),
            "confusion_matrix": confusion,
            "top_false_positives": false_positives.sort_values("spam_score", ascending=False)
            .head(10)[["id", "sender_domain", "subject", "spam_score", "spam_label", "ground_truth_label", "spam_reasons", "review_notes"]]
            .to_dict(orient="records"),
            "top_false_negatives": false_negatives.sort_values("spam_score", ascending=True)
            .head(10)[["id", "sender_domain", "subject", "spam_score", "spam_label", "ground_truth_label", "spam_reasons", "review_notes"]]
            .to_dict(orient="records"),
        }

    def _export_review_candidates(self, classified_frame: pd.DataFrame) -> None:
        review_export = classified_frame.sort_values("spam_score", ascending=False).head(200).copy()
        review_export = review_export.rename(columns={"id": "email_id", "spam_label": "predicted_label"})
        review_export["ground_truth_label"] = ""
        review_export["review_notes"] = ""
        review_export["reviewed_at"] = ""
        review_export[REVIEW_EXPORT_COLUMNS].to_csv(
            self.output_dir / "manual_review_candidates.csv",
            index=False,
        )

    def save_artifacts(
        self,
        classified_frame: pd.DataFrame,
        cluster_name_map: dict,
        top_terms_per_cluster: dict,
        evaluation_summary: dict,
        imported_review_count: int,
    ) -> dict:
        summary = {
            "total_emails": int(len(classified_frame)),
            "review_flagged": int((classified_frame["processed_flag"] == "review").sum()),
            "spam_label_counts": classified_frame["spam_label"].value_counts().to_dict(),
            "filters": {
                "start_date": self.start_date.strftime("%Y-%m-%d") if self.start_date is not None else None,
                "end_date": self.end_date.strftime("%Y-%m-%d") if self.end_date is not None else None,
                "incremental": self.incremental,
            },
            "imported_review_count": imported_review_count,
            "evaluation": evaluation_summary,
            "clusters": [],
            "spam_decision_rules": [
                "Authentication failures increase score the most, but trusted recurring senders receive smaller penalties.",
                "Promotional and phishing keywords increase score, with softer treatment for trusted transactional contexts.",
                "Routing complexity, aggressive punctuation, uppercase-heavy subjects, and unknown sender identity increase score.",
                "Sender frequency, stable sender-recipient history, and recurring authenticated domains reduce score as trust signals.",
                "Cluster context can add a small penalty when a message belongs to an unverified or marketing-heavy group.",
            ],
        }

        cluster_summary = (
            classified_frame.groupby(["cluster_id", "processed_category"])
            .agg(
                email_count=("id", "count"),
                dominant_domain=("sender_domain", lambda x: x.replace("", "unknown").mode().iloc[0] if not x.empty else "unknown"),
                avg_spf_pass=("spf_pass", "mean"),
                avg_dkim_pass=("dkim_pass", "mean"),
                avg_dmarc_pass=("dmarc_pass", "mean"),
                avg_spam_score=("spam_score", "mean"),
                trusted_sender_rate=("trusted_sender_profile", "mean"),
                likely_spam=("spam_label", lambda x: int((x == "likely_spam").sum())),
                suspicious=("spam_label", lambda x: int((x == "suspicious").sum())),
                likely_ham=("spam_label", lambda x: int((x == "likely_ham").sum())),
            )
            .reset_index()
            .sort_values("email_count", ascending=False)
        )

        for record in cluster_summary.to_dict(orient="records"):
            record["top_terms"] = top_terms_per_cluster.get(int(record["cluster_id"]), [])
            summary["clusters"].append(record)

        top_suspicious = classified_frame.sort_values("spam_score", ascending=False).head(15)
        summary["top_suspicious_examples"] = top_suspicious[
            ["id", "message_id", "sender_domain", "subject", "spam_score", "spam_label", "spam_reasons"]
        ].to_dict(orient="records")

        (self.output_dir / "classification_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (self.output_dir / "evaluation_summary.json").write_text(json.dumps(evaluation_summary, indent=2), encoding="utf-8")

        export_frame = classified_frame[
            [
                "id",
                "message_id",
                "sender_domain",
                "sender_email",
                "recipient_email",
                "subject",
                "processed_category",
                "processed_flag",
                "spam_score",
                "spam_label",
                "spam_reasons",
                "cluster_id",
                "domain_frequency",
                "sender_frequency",
                "sender_recipient_frequency",
                "prior_interaction",
                "stable_sender_recipient_pattern",
                "recurring_transactional_domain",
                "trusted_sender_profile",
                "trusted_free_mailbox_sender",
            ]
        ].copy()
        export_frame.to_csv(self.output_dir / "classified_emails.csv", index=False)

        self._export_review_candidates(classified_frame)
        self._plot_cluster_distribution(cluster_summary)
        self._plot_projection(classified_frame)
        self._plot_auth_rates(classified_frame)
        self._plot_top_domains(classified_frame)
        self._plot_spam_label_distribution(classified_frame)
        self._plot_spam_score_histogram(classified_frame)
        self._plot_spam_by_category(classified_frame)
        self._write_explanation(summary, cluster_name_map, top_terms_per_cluster)
        self._write_evaluation_report(evaluation_summary)
        return summary

    def _plot_cluster_distribution(self, cluster_summary: pd.DataFrame) -> None:
        plt.figure(figsize=(10, 6))
        plt.bar(cluster_summary["processed_category"], cluster_summary["email_count"], color="#355070")
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Email count")
        plt.title("Cluster distribution across mailbox")
        plt.tight_layout()
        plt.savefig(self.output_dir / "cluster_distribution.png", dpi=180)
        plt.close()

    def _plot_projection(self, classified_frame: pd.DataFrame) -> None:
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(
            classified_frame["component_1"],
            classified_frame["component_2"],
            c=classified_frame["cluster_id"],
            cmap="tab10",
            s=18,
            alpha=0.7,
        )
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.title("Mailbox clusters projected to two dimensions")
        plt.colorbar(scatter, label="Cluster ID")
        plt.tight_layout()
        plt.savefig(self.output_dir / "cluster_projection.png", dpi=180)
        plt.close()

    def _plot_auth_rates(self, classified_frame: pd.DataFrame) -> None:
        auth_summary = (
            classified_frame.groupby("processed_category")[["spf_pass", "dkim_pass", "dmarc_pass"]]
            .mean()
            .sort_values("spf_pass", ascending=False)
        )
        auth_summary.plot(kind="bar", figsize=(10, 6), color=["#6d597a", "#b56576", "#e56b6f"])
        plt.ylabel("Pass ratio")
        plt.ylim(0, 1)
        plt.title("Authentication pass rates by predicted category")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(self.output_dir / "auth_rates_by_category.png", dpi=180)
        plt.close()

    def _plot_top_domains(self, classified_frame: pd.DataFrame) -> None:
        top_domains = (
            classified_frame["sender_domain"]
            .replace("", "unknown")
            .value_counts()
            .head(10)
            .sort_values(ascending=True)
        )
        plt.figure(figsize=(10, 6))
        plt.barh(top_domains.index, top_domains.values, color="#43aa8b")
        plt.xlabel("Email count")
        plt.title("Top sender domains in the mailbox")
        plt.tight_layout()
        plt.savefig(self.output_dir / "top_sender_domains.png", dpi=180)
        plt.close()

    def _plot_spam_label_distribution(self, classified_frame: pd.DataFrame) -> None:
        counts = classified_frame["spam_label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
        plt.figure(figsize=(8, 5))
        plt.bar(counts.index, counts.values, color=["#4d908e", "#f9c74f", "#f94144"])
        plt.ylabel("Email count")
        plt.title("Spam classifier label distribution")
        plt.tight_layout()
        plt.savefig(self.output_dir / "spam_label_distribution.png", dpi=180)
        plt.close()

    def _plot_spam_score_histogram(self, classified_frame: pd.DataFrame) -> None:
        plt.figure(figsize=(9, 5))
        plt.hist(classified_frame["spam_score"], bins=20, color="#577590", edgecolor="white")
        plt.xlabel("Spam score")
        plt.ylabel("Email count")
        plt.title("Spam score distribution")
        plt.tight_layout()
        plt.savefig(self.output_dir / "spam_score_histogram.png", dpi=180)
        plt.close()

    def _plot_spam_by_category(self, classified_frame: pd.DataFrame) -> None:
        category_spam = (
            classified_frame.groupby(["processed_category", "spam_label"])["id"]
            .count()
            .unstack(fill_value=0)
            .reindex(columns=LABEL_ORDER, fill_value=0)
        )
        category_spam.plot(
            kind="bar",
            stacked=True,
            figsize=(10, 6),
            color=["#4d908e", "#f9c74f", "#f94144"],
        )
        plt.ylabel("Email count")
        plt.title("Spam labels within each predicted category")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(self.output_dir / "spam_by_category.png", dpi=180)
        plt.close()

    def _write_explanation(self, summary: dict, cluster_name_map: dict, top_terms_per_cluster: dict) -> None:
        suspicious_count = summary["spam_label_counts"].get("suspicious", 0)
        spam_count = summary["spam_label_counts"].get("likely_spam", 0)
        ham_count = summary["spam_label_counts"].get("likely_ham", 0)
        evaluation = summary["evaluation"]

        lines = [
            "# Classification And Spam Detection Overview",
            "",
            "## What The Pipeline Does",
            "",
            "1. Load email records from `emails.db`, with optional date filtering and incremental mode.",
            "2. Engineer text, authentication, routing, and trust-history features.",
            "3. Group similar emails with KMeans so the mailbox is segmented into behavioral categories.",
            "4. Name each cluster using its dominant domain, strongest subject keyword, and overall trust pattern.",
            "5. Run a deterministic spam scoring layer that now accounts for recurring trusted senders.",
            "6. Save the category, spam label, spam score, and spam reasons back into SQLite.",
            "7. Compare predictions against any manual review labels and write evaluation artifacts.",
            "",
            "## Decision Logic",
            "",
            "- Clustering remains unsupervised and groups similar emails without requiring a trained model.",
            "- `processed_category` explains the cluster the email belongs to.",
            "- `spam_score` is a 0-100 risk score built from weighted rules, not a trained probability.",
            "- `spam_label` is derived from `spam_score`: `likely_ham` < 40, `suspicious` 40-69.99, `likely_spam` >= 70.",
            "- `processed_flag` becomes `review` for both `suspicious` and `likely_spam` messages.",
            "",
            "## New Trust Signals",
            "",
            "- Sender frequency and sender-domain frequency reduce risk for recurring known senders.",
            "- Stable sender-recipient patterns reduce risk for personal mailbox conversations.",
            "- Recurring authenticated transactional domains soften keyword-based penalties.",
            "- Trusted free-mailbox senders are treated differently from unknown promotional free-mailbox senders.",
            "",
            "## Current Run",
            "",
            f"- Total emails analyzed: {summary['total_emails']}",
            f"- Likely ham: {ham_count}",
            f"- Suspicious: {suspicious_count}",
            f"- Likely spam: {spam_count}",
            f"- Review queue size: {summary['review_flagged']}",
            f"- Number of clusters: {len(cluster_name_map)}",
            f"- Imported review labels this run: {summary['imported_review_count']}",
            "",
            "## Evaluation Snapshot",
            "",
            f"- Reviewed emails: {evaluation['reviewed_count']}",
            f"- Exact match accuracy: {evaluation['exact_match_accuracy']}",
            f"- Review precision: {evaluation['review_precision']}",
            f"- Review recall: {evaluation['review_recall']}",
            "",
            "## Graph Guide",
            "",
            "- `cluster_distribution.png`: mailbox volume per predicted category.",
            "- `cluster_projection.png`: two-dimensional projection of all emails; each point is one email colored by cluster.",
            "- `auth_rates_by_category.png`: average SPF/DKIM/DMARC pass rates for each cluster.",
            "- `top_sender_domains.png`: top domains contributing mailbox traffic.",
            "- `spam_label_distribution.png`: how many emails were marked ham, suspicious, or spam.",
            "- `spam_score_histogram.png`: where scores concentrate across the full mailbox.",
            "- `spam_by_category.png`: stacked view of spam labels within each predicted category.",
            "",
            "## Cluster Notes",
            "",
        ]

        for cluster in summary["clusters"]:
            cluster_id = int(cluster["cluster_id"])
            lines.append(
                f"- `{cluster['processed_category']}`: {cluster['email_count']} emails, dominant domain `{cluster['dominant_domain']}`, "
                f"avg spam score `{cluster['avg_spam_score']:.2f}`, trusted sender rate `{cluster['trusted_sender_rate']:.2f}`, "
                f"spam `{cluster['likely_spam']}`, suspicious `{cluster['suspicious']}`, top terms {top_terms_per_cluster.get(cluster_id, [])}."
            )

        lines.extend(["", "## Highest-Risk Examples", ""])
        for record in summary["top_suspicious_examples"][:10]:
            lines.append(
                f"- Row `{record['id']}` from `{record['sender_domain']}` scored `{record['spam_score']}` as `{record['spam_label']}` "
                f"for subject `{record['subject']}` because: {record['spam_reasons']}."
            )

        (self.output_dir / "classification_explanation.md").write_text("\n".join(lines), encoding="utf-8")

    def _write_evaluation_report(self, evaluation_summary: dict) -> None:
        lines = [
            "# Evaluation Report",
            "",
            "## Overview",
            "",
            f"- Reviewed emails: {evaluation_summary['reviewed_count']}",
            f"- Exact match accuracy: {evaluation_summary['exact_match_accuracy']}",
            f"- Review precision: {evaluation_summary['review_precision']}",
            f"- Review recall: {evaluation_summary['review_recall']}",
            f"- Likely spam precision: {evaluation_summary['likely_spam_precision']}",
            f"- Likely spam recall: {evaluation_summary['likely_spam_recall']}",
            "",
            "## Confusion Matrix",
            "",
        ]

        if evaluation_summary["confusion_matrix"]:
            for row in evaluation_summary["confusion_matrix"]:
                lines.append(
                    f"- actual `{row['actual_label']}` vs predicted `{row['predicted_label']}`: {row['count']}"
                )
        else:
            lines.append("- No manual review labels were available for evaluation.")

        lines.extend(["", "## Top False Positives", ""])
        if evaluation_summary["top_false_positives"]:
            for record in evaluation_summary["top_false_positives"]:
                lines.append(
                    f"- Row `{record['id']}` predicted `{record['spam_label']}` but reviewed as `{record['ground_truth_label']}` "
                    f"with score `{record['spam_score']}` for subject `{record['subject']}`. Notes: {record['review_notes'] or 'n/a'}."
                )
        else:
            lines.append("- No reviewed false positives yet.")

        lines.extend(["", "## Top False Negatives", ""])
        if evaluation_summary["top_false_negatives"]:
            for record in evaluation_summary["top_false_negatives"]:
                lines.append(
                    f"- Row `{record['id']}` predicted `{record['spam_label']}` but reviewed as `{record['ground_truth_label']}` "
                    f"with score `{record['spam_score']}` for subject `{record['subject']}`. Notes: {record['review_notes'] or 'n/a'}."
                )
        else:
            lines.append("- No reviewed false negatives yet.")

        (self.output_dir / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")

    def _write_empty_run_artifacts(self, imported_review_count: int) -> dict:
        evaluation_summary = {
            "reviewed_count": 0,
            "exact_match_accuracy": None,
            "review_precision": None,
            "review_recall": None,
            "likely_spam_precision": None,
            "likely_spam_recall": None,
            "confusion_matrix": [],
            "top_false_positives": [],
            "top_false_negatives": [],
        }
        summary = {
            "status": "no_new_emails",
            "message": "No emails matched the current filters. Incremental mode found no unclassified emails.",
            "total_emails": 0,
            "review_flagged": 0,
            "spam_label_counts": {},
            "filters": {
                "start_date": self.start_date.strftime("%Y-%m-%d") if self.start_date is not None else None,
                "end_date": self.end_date.strftime("%Y-%m-%d") if self.end_date is not None else None,
                "incremental": self.incremental,
            },
            "imported_review_count": imported_review_count,
            "evaluation": evaluation_summary,
            "clusters": [],
            "top_suspicious_examples": [],
            "spam_decision_rules": [],
        }
        (self.output_dir / "classification_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (self.output_dir / "evaluation_summary.json").write_text(json.dumps(evaluation_summary, indent=2), encoding="utf-8")
        (self.output_dir / "classified_emails.csv").write_text(
            "id,message_id,sender_domain,sender_email,recipient_email,subject,processed_category,processed_flag,spam_score,spam_label,spam_reasons,cluster_id,domain_frequency,sender_frequency,sender_recipient_frequency,prior_interaction,stable_sender_recipient_pattern,recurring_transactional_domain,trusted_sender_profile,trusted_free_mailbox_sender\n",
            encoding="utf-8",
        )
        (self.output_dir / "manual_review_candidates.csv").write_text(
            ",".join(REVIEW_EXPORT_COLUMNS) + "\n",
            encoding="utf-8",
        )
        self._write_evaluation_report(evaluation_summary)
        return summary

    def run(self) -> dict:
        frame = self.load_data()
        if frame.empty:
            imported_review_count = self.import_review_csv()
            if self.incremental:
                logging.info("Incremental classification found no unclassified emails to process")
                return self._write_empty_run_artifacts(imported_review_count)
            raise ValueError("No emails found in the database for classification with the current filters.")

        engineered = self.engineer_features(frame)
        feature_matrix, vectorizer = self.build_feature_matrix(engineered)
        clustered_frame, cluster_name_map, top_terms_per_cluster = self.assign_clusters(
            engineered,
            feature_matrix,
            vectorizer,
        )
        classified_frame = self.score_emails(clustered_frame)
        component_1, component_2 = self._project_clusters(feature_matrix, len(classified_frame))
        classified_frame["component_1"] = component_1
        classified_frame["component_2"] = component_2

        self.persist_results(classified_frame)
        imported_review_count = self.import_review_csv()
        evaluation_summary = self.evaluate_predictions(classified_frame)
        summary = self.save_artifacts(
            classified_frame,
            cluster_name_map,
            top_terms_per_cluster,
            evaluation_summary,
            imported_review_count,
        )
        logging.info("Classification finished for %s emails", len(classified_frame))
        return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OpenDeMail classification and evaluation.")
    parser.add_argument("--db-path", default="emails.db", help="Path to the SQLite database.")
    parser.add_argument("--output-dir", default="classification_output", help="Directory for generated artifacts.")
    parser.add_argument(
        "--review-csv",
        default=None,
        help="Optional CSV containing manual review labels with email_id and ground_truth_label columns.",
    )
    parser.add_argument("--start-date", default=None, help="Optional start date filter in YYYY-MM-DD format.")
    parser.add_argument("--end-date", default=None, help="Optional end date filter in YYYY-MM-DD format.")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only classify emails that do not yet have stored classification results.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    classifier = EmailClassifier(
        db_path=args.db_path,
        output_dir=args.output_dir,
        review_csv_path=args.review_csv,
        start_date=args.start_date,
        end_date=args.end_date,
        incremental=args.incremental,
    )
    summary = classifier.run()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
