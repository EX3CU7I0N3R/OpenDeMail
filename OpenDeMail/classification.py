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

    def prepare_features(self, frame: pd.DataFrame):
        working = frame.copy()
        working["sender_domain"] = working["sender_domain"].fillna("").str.lower()
        working["sender_email"] = working["sender_email"].fillna("").str.lower()
        working["subject"] = working["subject"].fillna("")
        working["mailer"] = working["mailer"].fillna("")
        working["content_type"] = working["content_type"].fillna("")
        working["spf_result"] = working["spf_result"].fillna("none").str.lower()
        working["dkim_result"] = working["dkim_result"].fillna("none").str.lower()
        working["dmarc_result"] = working["dmarc_result"].fillna("none").str.lower()
        working["received"] = working["received"].fillna("")
        working["x_received"] = working["x_received"].fillna("")

        working["subject_token_count"] = working["subject"].str.split().str.len().fillna(0)
        working["received_hops"] = working["received"].str.count("\n").fillna(0) + 1
        working.loc[working["received"].eq(""), "received_hops"] = 0
        working["x_received_hops"] = working["x_received"].str.count("\n").fillna(0) + 1
        working.loc[working["x_received"].eq(""), "x_received_hops"] = 0
        working["spf_pass"] = working["spf_result"].eq("pass").astype(int)
        working["dkim_pass"] = working["dkim_result"].eq("pass").astype(int)
        working["dmarc_pass"] = working["dmarc_result"].eq("pass").astype(int)

        suspicious_terms = re.compile(
            r"\b(verify|urgent|winner|offer|save|free|discount|limited|sale|special|alert|deal)\b",
            re.IGNORECASE,
        )
        working["promo_term_hits"] = working["subject"].str.count(suspicious_terms).fillna(0)

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

        subjects = " ".join(cluster_frame["subject"].astype(str).tolist()).lower()
        tokens = [
            token
            for token in re.findall(r"[a-z]{4,}", subjects)
            if token not in {"from", "with", "your", "this", "have", "that", "today", "email"}
        ]
        token_hint = pd.Series(tokens).value_counts().head(1)
        keyword = token_hint.index[0] if not token_hint.empty else "general"

        pass_rate = (
            cluster_frame[["spf_pass", "dkim_pass", "dmarc_pass"]].mean().mean()
            if len(cluster_frame) > 0
            else 0.0
        )
        promo_rate = cluster_frame["promo_term_hits"].gt(0).mean() if len(cluster_frame) > 0 else 0.0

        if pass_rate < 0.45:
            prefix = "unverified"
        elif promo_rate > 0.4:
            prefix = "marketing"
        else:
            prefix = "transactional"
        return f"{prefix}_{top_domain}_{keyword}"

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

        working["risk_score"] = (
            (1 - working["spf_pass"])
            + (1 - working["dkim_pass"])
            + (1 - working["dmarc_pass"])
            + working["received_hops"].ge(4).astype(int)
            + working["promo_term_hits"].ge(2).astype(int)
        )
        working["processed_flag"] = np.where(
            working["risk_score"] >= 3,
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
            updates = classified_frame[["id", "processed_category", "processed_flag"]].itertuples(index=False, name=None)
            db.bulk_update_classification(updates)
        finally:
            db.close()

    def save_artifacts(self, classified_frame: pd.DataFrame, cluster_name_map: dict, top_terms_per_cluster: dict) -> dict:
        summary = {
            "total_emails": int(len(classified_frame)),
            "clusters": [],
            "review_flagged": int((classified_frame["processed_flag"] == "review").sum()),
        }

        cluster_summary = (
            classified_frame.groupby(["cluster_id", "processed_category"])
            .agg(
                email_count=("id", "count"),
                dominant_domain=("sender_domain", lambda x: x.replace("", "unknown").mode().iloc[0] if not x.empty else "unknown"),
                avg_spf_pass=("spf_pass", "mean"),
                avg_dkim_pass=("dkim_pass", "mean"),
                avg_dmarc_pass=("dmarc_pass", "mean"),
                flagged_for_review=("processed_flag", lambda x: int((x == "review").sum())),
            )
            .reset_index()
            .sort_values("email_count", ascending=False)
        )

        for record in cluster_summary.to_dict(orient="records"):
            record["top_terms"] = top_terms_per_cluster.get(int(record["cluster_id"]), [])
            summary["clusters"].append(record)

        summary_path = self.output_dir / "classification_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        top_records = classified_frame[
            [
                "id",
                "message_id",
                "sender_domain",
                "subject",
                "processed_category",
                "processed_flag",
                "risk_score",
                "cluster_id",
            ]
        ].copy()
        top_records.to_csv(self.output_dir / "classified_emails.csv", index=False)

        self._plot_cluster_distribution(cluster_summary)
        self._plot_projection(classified_frame)
        self._plot_auth_rates(classified_frame)
        self._plot_top_domains(classified_frame)
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

    def _write_explanation(self, summary: dict, cluster_name_map: dict, top_terms_per_cluster: dict) -> None:
        lines = [
            "# Classification Overview",
            "",
            f"- Total emails analyzed: {summary['total_emails']}",
            f"- Emails flagged for review: {summary['review_flagged']}",
            f"- Number of predicted clusters: {len(cluster_name_map)}",
            "",
            "## Graph Guide",
            "",
            "- `cluster_distribution.png`: shows how the mailbox splits across predicted categories. Taller bars indicate the dominant email behavior groups.",
            "- `cluster_projection.png`: shows each email compressed into two dimensions. Tight groups suggest coherent patterns; overlap suggests similar message types.",
            "- `auth_rates_by_category.png`: compares SPF, DKIM, and DMARC pass ratios by category. Lower bars indicate weaker sender verification.",
            "- `top_sender_domains.png`: highlights which domains contribute the most traffic overall.",
            "",
            "## Cluster Notes",
            "",
        ]

        for cluster in summary["clusters"]:
            cluster_id = int(cluster["cluster_id"])
            lines.append(
                f"- `{cluster['processed_category']}`: {cluster['email_count']} emails, dominant domain `{cluster['dominant_domain']}`, "
                f"review-flagged `{cluster['flagged_for_review']}`, top terms {top_terms_per_cluster.get(cluster_id, [])}."
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
