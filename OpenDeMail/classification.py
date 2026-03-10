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
FREE_MAILBOX_DOMAINS = {"gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "live.com"}


class EmailClassifier:
    def __init__(self, db_path: str = "emails.db", output_dir: str = "classification_output") -> None:
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        db = MailDB(self.db_path)
        try:
            query = """
                SELECT
                    id,
                    message_id,
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
                    raw_headers_json
                FROM emails
            """
            frame = pd.read_sql_query(query, db.conn)
            logging.info("Loaded %s emails for classification", len(frame))
            return frame
        finally:
            db.close()

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

    def prepare_features(self, frame: pd.DataFrame):
        working = frame.copy()
        for column in [
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
        working["promo_term_hits"] = working["subject"].str.count(PROMO_PATTERN).fillna(0)
        working["phishing_term_hits"] = working["subject"].str.count(PHISHING_PATTERN).fillna(0)
        working["exclamation_count"] = working["subject"].str.count("!").fillna(0)
        working["uppercase_ratio"] = working["subject"].map(
            lambda text: (
                sum(1 for char in str(text) if char.isupper()) / max(sum(1 for char in str(text) if char.isalpha()), 1)
            )
        )
        working["is_html"] = working["content_type"].str.contains("html", case=False, regex=False).astype(int)
        working["free_mailbox_sender"] = working["sender_domain"].isin(FREE_MAILBOX_DOMAINS).astype(int)
        working["unknown_sender_domain"] = working["sender_domain"].eq("").astype(int)

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
                    "exclamation_count",
                    "uppercase_ratio",
                    "is_html",
                    "free_mailbox_sender",
                    "unknown_sender_domain",
                ]
            ].to_numpy(dtype=float)
        )
        feature_matrix = hstack([text_features, numeric_features]).tocsr()
        return working, feature_matrix, vectorizer

    @staticmethod
    def _select_cluster_count(feature_matrix) -> int:
        sample_size = feature_matrix.shape[0]
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

        if pass_rate < 0.45 or phishing_rate > 0.30:
            prefix = "unverified"
        elif promo_rate > 0.40:
            prefix = "marketing"
        else:
            prefix = "transactional"
        return f"{prefix}_{top_domain}_{keyword}"

    @staticmethod
    def _score_single_email(row: pd.Series) -> tuple[float, str, str]:
        score = 0.0
        reasons = []

        if not row["spf_pass"]:
            score += 1.4
            reasons.append("SPF did not pass")
        if not row["dkim_pass"]:
            score += 1.4
            reasons.append("DKIM did not pass")
        if not row["dmarc_pass"]:
            score += 1.2
            reasons.append("DMARC did not pass")
        if row["promo_term_hits"] > 0:
            score += min(1.8, 0.6 * float(row["promo_term_hits"]))
            reasons.append(f"Promotional language detected ({int(row['promo_term_hits'])} hit(s))")
        if row["phishing_term_hits"] > 0:
            score += min(2.1, 0.7 * float(row["phishing_term_hits"]))
            reasons.append(f"Security/payment keywords detected ({int(row['phishing_term_hits'])} hit(s))")
        if row["received_hops"] >= 4:
            score += 0.6
            reasons.append("High number of mail transfer hops")
        if row["exclamation_count"] >= 2:
            score += 0.5
            reasons.append("Aggressive punctuation in subject")
        if row["uppercase_ratio"] >= 0.45 and row["subject_token_count"] >= 3:
            score += 0.5
            reasons.append("High uppercase ratio in subject")
        if row["free_mailbox_sender"] and row["promo_term_hits"] > 0:
            score += 0.6
            reasons.append("Free mailbox domain used with promotional subject")
        if row["unknown_sender_domain"]:
            score += 0.7
            reasons.append("Sender domain missing")
        if row["is_html"] and row["promo_term_hits"] > 0:
            score += 0.4
            reasons.append("HTML email combined with promotional content")
        if str(row["processed_category"]).startswith("unverified_"):
            score += 0.8
            reasons.append("Cluster behaves like an unverified sender group")
        elif str(row["processed_category"]).startswith("marketing_"):
            score += 0.4
            reasons.append("Cluster behaves like a marketing sender group")

        normalized_score = round(min(100.0, (score / 8.5) * 100.0), 2)
        if normalized_score >= 70:
            label = "likely_spam"
        elif normalized_score >= 40:
            label = "suspicious"
        else:
            label = "likely_ham"

        if not reasons:
            reasons.append("No strong spam indicators were triggered")
        return normalized_score, label, "; ".join(reasons)

    def classify(self, frame: pd.DataFrame):
        working, feature_matrix, vectorizer = self.prepare_features(frame)
        cluster_count = self._select_cluster_count(feature_matrix)
        model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
        labels = model.fit_predict(feature_matrix)

        working["cluster_id"] = labels
        cluster_name_map = {}
        for cluster_id in sorted(working["cluster_id"].unique()):
            cluster_name_map[cluster_id] = self._summarize_cluster_name(working[working["cluster_id"] == cluster_id])

        working["processed_category"] = working["cluster_id"].map(cluster_name_map)

        spam_results = working.apply(self._score_single_email, axis=1, result_type="expand")
        spam_results.columns = ["spam_score", "spam_label", "spam_reasons"]
        working[["spam_score", "spam_label", "spam_reasons"]] = spam_results
        working["processed_flag"] = np.where(
            working["spam_label"].isin(["likely_spam", "suspicious"]),
            "review",
            "normal",
        )

        svd = TruncatedSVD(n_components=2, random_state=42)
        projection = svd.fit_transform(feature_matrix)
        working["component_1"] = projection[:, 0]
        working["component_2"] = projection[:, 1]

        top_terms_per_cluster = {}
        feature_names = np.array(vectorizer.get_feature_names_out())
        centroids = model.cluster_centers_[:, : len(feature_names)]
        for cluster_id, centroid in enumerate(centroids):
            top_terms_per_cluster[int(cluster_id)] = feature_names[np.argsort(centroid)[-8:][::-1]].tolist()

        return working, cluster_name_map, top_terms_per_cluster

    def persist_results(self, classified_frame: pd.DataFrame) -> None:
        db = MailDB(self.db_path)
        try:
            updates = classified_frame[
                ["id", "processed_category", "processed_flag", "spam_score", "spam_label", "spam_reasons"]
            ].itertuples(index=False, name=None)
            db.bulk_update_classification(updates)
        finally:
            db.close()

    def save_artifacts(self, classified_frame: pd.DataFrame, cluster_name_map: dict, top_terms_per_cluster: dict) -> dict:
        summary = {
            "total_emails": int(len(classified_frame)),
            "review_flagged": int((classified_frame["processed_flag"] == "review").sum()),
            "spam_label_counts": classified_frame["spam_label"].value_counts().to_dict(),
            "clusters": [],
            "spam_decision_rules": [
                "Authentication failures increase score the most.",
                "Promotional and phishing keywords increase score.",
                "Routing complexity, aggressive punctuation, uppercase-heavy subjects, and unknown sender identity increase score.",
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

        export_frame = classified_frame[
            [
                "id",
                "message_id",
                "sender_domain",
                "subject",
                "processed_category",
                "processed_flag",
                "spam_score",
                "spam_label",
                "spam_reasons",
                "cluster_id",
            ]
        ].copy()
        export_frame.to_csv(self.output_dir / "classified_emails.csv", index=False)

        self._plot_cluster_distribution(cluster_summary)
        self._plot_projection(classified_frame)
        self._plot_auth_rates(classified_frame)
        self._plot_top_domains(classified_frame)
        self._plot_spam_label_distribution(classified_frame)
        self._plot_spam_score_histogram(classified_frame)
        self._plot_spam_by_category(classified_frame)
        self._write_explanation(summary, cluster_name_map, top_terms_per_cluster)
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
        counts = classified_frame["spam_label"].value_counts().reindex(["likely_ham", "suspicious", "likely_spam"], fill_value=0)
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
            .reindex(columns=["likely_ham", "suspicious", "likely_spam"], fill_value=0)
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

        lines = [
            "# Classification And Spam Detection Overview",
            "",
            "## What The Pipeline Does",
            "",
            "1. Load every email header record from `emails.db`.",
            "2. Build clustering features from subject text, sender identity, content type, authentication results, and routing depth.",
            "3. Group similar emails with KMeans so the mailbox is segmented into behavioral categories.",
            "4. Name each cluster using its dominant domain, strongest subject keyword, and overall trust pattern.",
            "5. Run a spam scoring layer on top of each email using deterministic risk rules.",
            "6. Save the category, spam label, spam score, and spam reasons back into SQLite.",
            "",
            "## Decision Logic",
            "",
            "- Clustering is unsupervised. It groups similar emails without using manual labels.",
            "- `processed_category` explains the cluster the email belongs to.",
            "- `spam_score` is a 0-100 risk score built from weighted rules, not a trained probability.",
            "- `spam_label` is derived from `spam_score`: `likely_ham` < 40, `suspicious` 40-69.99, `likely_spam` >= 70.",
            "- `processed_flag` becomes `review` for both `suspicious` and `likely_spam` messages.",
            "",
            "## Main Spam Signals",
            "",
            "- Authentication failures: SPF, DKIM, and DMARC misses carry the heaviest penalties.",
            "- Subject language: promotional words, security/payment terms, excessive exclamation, and uppercase-heavy subjects raise the score.",
            "- Routing shape: many mail hops add risk because they often indicate forwarding chains or indirect delivery.",
            "- Sender identity: missing sender domain or a free mailbox paired with promotional language adds risk.",
            "- Cluster context: messages inside an `unverified_*` or `marketing_*` cluster receive a small extra penalty.",
            "",
            "## Current Run",
            "",
            f"- Total emails analyzed: {summary['total_emails']}",
            f"- Likely ham: {ham_count}",
            f"- Suspicious: {suspicious_count}",
            f"- Likely spam: {spam_count}",
            f"- Review queue size: {summary['review_flagged']}",
            f"- Number of clusters: {len(cluster_name_map)}",
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
                f"avg spam score `{cluster['avg_spam_score']:.2f}`, spam `{cluster['likely_spam']}`, suspicious `{cluster['suspicious']}`, "
                f"top terms {top_terms_per_cluster.get(cluster_id, [])}."
            )

        lines.extend(["", "## Highest-Risk Examples", ""])
        for record in summary["top_suspicious_examples"][:10]:
            lines.append(
                f"- Row `{record['id']}` from `{record['sender_domain']}` scored `{record['spam_score']}` as `{record['spam_label']}` "
                f"for subject `{record['subject']}` because: {record['spam_reasons']}."
            )

        (self.output_dir / "classification_explanation.md").write_text("\n".join(lines), encoding="utf-8")

    def run(self) -> dict:
        frame = self.load_data()
        if frame.empty:
            raise ValueError("No emails found in the database for classification.")

        classified_frame, cluster_name_map, top_terms_per_cluster = self.classify(frame)
        self.persist_results(classified_frame)
        summary = self.save_artifacts(classified_frame, cluster_name_map, top_terms_per_cluster)
        logging.info("Classification finished for %s emails", len(classified_frame))
        return summary


def main() -> int:
    classifier = EmailClassifier()
    summary = classifier.run()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
