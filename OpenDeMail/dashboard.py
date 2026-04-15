import json
from pathlib import Path

import pandas as pd
import streamlit as st

from OpenDeMail.modules.mailDB import MailDB


DEFAULT_DB_PATH = "emails.db"
DEFAULT_OUTPUT_DIR = "classification_output"
LABEL_OPTIONS = ["likely_ham", "suspicious", "likely_spam"]
STATUS_COLORS = {
    "likely_ham": "#247a5a",
    "suspicious": "#c68b17",
    "likely_spam": "#b83232",
}


def configure_page() -> None:
    st.set_page_config(
        page_title="OpenDeMail Dashboard",
        page_icon="📬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(243, 211, 167, 0.22), transparent 32%),
                radial-gradient(circle at top left, rgba(93, 130, 110, 0.18), transparent 28%),
                linear-gradient(180deg, #f7f3ea 0%, #efe6d6 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            color: #1f2a24;
            letter-spacing: -0.02em;
        }
        .metric-card {
            border: 1px solid rgba(31, 42, 36, 0.08);
            border-radius: 18px;
            background: rgba(255, 252, 246, 0.9);
            padding: 1rem 1.1rem;
            box-shadow: 0 18px 40px rgba(81, 61, 41, 0.08);
        }
        .section-card {
            border: 1px solid rgba(31, 42, 36, 0.08);
            border-radius: 22px;
            background: rgba(255, 252, 246, 0.94);
            padding: 1.25rem;
            box-shadow: 0 18px 40px rgba(81, 61, 41, 0.08);
        }
        .small-muted {
            color: #5f6c64;
            font-size: 0.92rem;
        }
        .pill {
            display: inline-block;
            border-radius: 999px;
            padding: 0.2rem 0.65rem;
            font-size: 0.8rem;
            font-weight: 700;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
            background: #f1eadf;
            color: #34423a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_summary(output_dir: str) -> dict:
    summary_path = Path(output_dir) / "classification_summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_evaluation_summary(output_dir: str) -> dict:
    summary_path = Path(output_dir) / "evaluation_summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_email_frame(db_path: str) -> pd.DataFrame:
    db = MailDB(db_path)
    try:
        query = """
            SELECT
                e.id,
                e.message_id,
                e.sender_name,
                e.sender_email,
                e.sender_domain,
                e.recipient_email,
                e.subject,
                e.date_header,
                e.processed_category,
                e.processed_flag,
                e.spam_score,
                e.spam_label,
                e.spam_reasons,
                e.spf_result,
                e.dkim_result,
                e.dmarc_result,
                e.content_type,
                e.mailer,
                e.received,
                e.x_received,
                e.raw_headers_json,
                e.created_at,
                r.ground_truth_label,
                r.review_notes,
                r.reviewed_at
            FROM emails e
            LEFT JOIN review_labels r ON r.email_id = e.id
            ORDER BY e.id DESC
        """
        frame = pd.read_sql_query(query, db.conn)
    finally:
        db.close()

    if frame.empty:
        return frame

    frame["spam_score"] = pd.to_numeric(frame["spam_score"], errors="coerce")
    frame["parsed_date"] = pd.to_datetime(frame["date_header"], errors="coerce", utc=True)
    frame["date_label"] = frame["parsed_date"].dt.strftime("%Y-%m-%d %H:%M").fillna("Unknown date")
    frame["subject"] = frame["subject"].fillna("").replace("", "(no subject)")
    frame["sender_domain"] = frame["sender_domain"].fillna("").replace("", "unknown")
    frame["sender_email"] = frame["sender_email"].fillna("").replace("", "unknown")
    frame["processed_category"] = frame["processed_category"].fillna("").replace("", "unclassified")
    frame["spam_label"] = frame["spam_label"].fillna("").replace("", "unclassified")
    frame["processed_flag"] = frame["processed_flag"].fillna("").replace("", "pending")
    frame["review_status"] = frame["ground_truth_label"].fillna("").replace("", "unreviewed")
    return frame


def compute_live_evaluation(email_frame: pd.DataFrame) -> dict:
    reviewed = email_frame[email_frame["ground_truth_label"].notna() & email_frame["ground_truth_label"].ne("")].copy()
    if reviewed.empty:
        return {
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

    review_predicted = reviewed["spam_label"].isin(["suspicious", "likely_spam"])
    review_actual = reviewed["ground_truth_label"].isin(["suspicious", "likely_spam"])
    spam_predicted = reviewed["spam_label"].eq("likely_spam")
    spam_actual = reviewed["ground_truth_label"].eq("likely_spam")

    confusion = []
    confusion_table = pd.crosstab(reviewed["ground_truth_label"], reviewed["spam_label"], dropna=False)
    for actual_label in LABEL_OPTIONS:
        for predicted_label in LABEL_OPTIONS:
            confusion.append(
                {
                    "actual_label": actual_label,
                    "predicted_label": predicted_label,
                    "count": int(confusion_table.get(predicted_label, pd.Series(dtype=int)).get(actual_label, 0)),
                }
            )

    def safe_ratio(numerator: int, denominator: int) -> float | None:
        if denominator == 0:
            return None
        return round(float(numerator) / float(denominator), 4)

    false_positives = reviewed[(review_predicted) & (reviewed["ground_truth_label"] == "likely_ham")]
    false_negatives = reviewed[(~review_predicted) & (review_actual)]

    return {
        "reviewed_count": int(len(reviewed)),
        "exact_match_accuracy": round(float((reviewed["ground_truth_label"] == reviewed["spam_label"]).mean()), 4),
        "review_precision": safe_ratio(int((review_predicted & review_actual).sum()), int(review_predicted.sum())),
        "review_recall": safe_ratio(int((review_predicted & review_actual).sum()), int(review_actual.sum())),
        "likely_spam_precision": safe_ratio(int((spam_predicted & spam_actual).sum()), int(spam_predicted.sum())),
        "likely_spam_recall": safe_ratio(int((spam_predicted & spam_actual).sum()), int(spam_actual.sum())),
        "confusion_matrix": confusion,
        "top_false_positives": false_positives.sort_values("spam_score", ascending=False)
        .head(10)[["id", "subject", "spam_score", "spam_label", "ground_truth_label", "review_notes"]]
        .to_dict(orient="records"),
        "top_false_negatives": false_negatives.sort_values("spam_score", ascending=True)
        .head(10)[["id", "subject", "spam_score", "spam_label", "ground_truth_label", "review_notes"]]
        .to_dict(orient="records"),
    }


def save_review_label(
    db_path: str,
    email_id: int,
    ground_truth_label: str,
    review_notes: str,
    reviewed_at: str,
) -> None:
    db = MailDB(db_path)
    try:
        db.bulk_upsert_manual_reviews([(email_id, ground_truth_label, review_notes, reviewed_at)])
    finally:
        db.close()
    load_email_frame.clear()


def metric_card(label: str, value: str, help_text: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="small-muted">{label}</div>
            <div style="font-size:1.9rem;font-weight:800;color:#1f2a24;">{value}</div>
            <div class="small-muted">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_score(value: object) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.2f}"


def render_header(summary: dict, email_frame: pd.DataFrame) -> None:
    total_rows = len(email_frame)
    reviewed_rows = int(email_frame["ground_truth_label"].notna().sum()) if not email_frame.empty else 0
    review_queue = int(email_frame["processed_flag"].eq("review").sum()) if not email_frame.empty else 0
    st.title("OpenDeMail Intelligence Dashboard")
    st.caption(
        "Interactive triage for clustering, spam scoring, evaluation, and manual review on top of your SQLite mailbox."
    )
    cols = st.columns(4)
    with cols[0]:
        metric_card("Mailbox Rows", str(total_rows), "Total email records currently available in the database.")
    with cols[1]:
        metric_card("Review Queue", str(review_queue), "Emails currently marked for human review.")
    with cols[2]:
        metric_card("Reviewed Labels", str(reviewed_rows), "Ground-truth labels stored in the review table.")
    with cols[3]:
        status_value = summary.get("status", "active")
        metric_card("Latest Run", status_value.replace("_", " ").title(), "Status from the latest saved classification summary.")


def render_sidebar(email_frame: pd.DataFrame) -> dict:
    st.sidebar.header("Filters")
    query = st.sidebar.text_input("Search subject, sender, or reason")

    labels = sorted(email_frame["spam_label"].dropna().unique().tolist()) if not email_frame.empty else []
    selected_labels = st.sidebar.multiselect("Spam label", labels, default=labels)

    categories = sorted(email_frame["processed_category"].dropna().unique().tolist()) if not email_frame.empty else []
    default_categories = categories[:10] if len(categories) > 10 else categories
    selected_categories = st.sidebar.multiselect("Cluster category", categories, default=default_categories)

    review_statuses = sorted(email_frame["review_status"].dropna().unique().tolist()) if not email_frame.empty else []
    selected_review_statuses = st.sidebar.multiselect(
        "Review status",
        review_statuses,
        default=review_statuses,
    )

    min_score, max_score = 0.0, 100.0
    if not email_frame.empty and email_frame["spam_score"].notna().any():
        min_score = float(email_frame["spam_score"].min())
        max_score = float(email_frame["spam_score"].max())
    score_range = st.sidebar.slider(
        "Spam score range",
        min_value=0.0,
        max_value=100.0,
        value=(max(0.0, min_score), min(100.0, max_score if max_score > 0 else 100.0)),
        step=1.0,
    )

    return {
        "query": query.strip().lower(),
        "labels": selected_labels,
        "categories": selected_categories,
        "review_statuses": selected_review_statuses,
        "score_range": score_range,
    }


def apply_filters(email_frame: pd.DataFrame, filters: dict) -> pd.DataFrame:
    if email_frame.empty:
        return email_frame

    filtered = email_frame.copy()
    if filters["query"]:
        search_blob = (
            filtered["subject"].fillna("")
            + " "
            + filtered["sender_email"].fillna("")
            + " "
            + filtered["sender_domain"].fillna("")
            + " "
            + filtered["spam_reasons"].fillna("")
        ).str.lower()
        filtered = filtered[search_blob.str.contains(filters["query"], regex=False)]

    if filters["labels"]:
        filtered = filtered[filtered["spam_label"].isin(filters["labels"])]
    if filters["categories"]:
        filtered = filtered[filtered["processed_category"].isin(filters["categories"])]
    if filters["review_statuses"]:
        filtered = filtered[filtered["review_status"].isin(filters["review_statuses"])]

    score_low, score_high = filters["score_range"]
    filtered = filtered[
        filtered["spam_score"].fillna(0).between(score_low, score_high, inclusive="both")
    ]
    return filtered


def render_overview_tab(summary: dict, evaluation_summary: dict, filtered_frame: pd.DataFrame) -> None:
    left, right = st.columns([1.4, 1.1])
    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Mailbox Health")
        label_counts = (
            filtered_frame["spam_label"].value_counts().reindex(LABEL_OPTIONS, fill_value=0)
            if not filtered_frame.empty
            else pd.Series([0, 0, 0], index=LABEL_OPTIONS)
        )
        st.bar_chart(label_counts)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Top Sender Domains")
        top_domains = (
            filtered_frame["sender_domain"].value_counts().head(10).sort_values(ascending=False)
            if not filtered_frame.empty
            else pd.Series(dtype=int)
        )
        if top_domains.empty:
            st.info("No sender-domain data is available for the active filter set.")
        else:
            st.dataframe(
                top_domains.rename_axis("sender_domain").reset_index(name="email_count"),
                use_container_width=True,
                hide_index=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Latest Evaluation")
        reviewed_count = evaluation_summary.get("reviewed_count", 0)
        if reviewed_count:
            eval_cols = st.columns(2)
            with eval_cols[0]:
                st.metric("Exact Match Accuracy", evaluation_summary.get("exact_match_accuracy"))
                st.metric("Review Precision", evaluation_summary.get("review_precision"))
            with eval_cols[1]:
                st.metric("Review Recall", evaluation_summary.get("review_recall"))
                st.metric("Likely Spam Precision", evaluation_summary.get("likely_spam_precision"))
        else:
            st.info("No reviewed labels are available yet. Use the Review Lab tab to add them.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Run Metadata")
        filters = summary.get("filters", {})
        st.write(
            {
                "status": summary.get("status", "active"),
                "start_date": filters.get("start_date"),
                "end_date": filters.get("end_date"),
                "incremental": filters.get("incremental"),
                "imported_review_count": summary.get("imported_review_count", 0),
            }
        )
        st.markdown("</div>", unsafe_allow_html=True)


def render_queue_tab(filtered_frame: pd.DataFrame) -> None:
    st.subheader("Suspicious Queue")
    queue_frame = filtered_frame[filtered_frame["processed_flag"] == "review"].copy()
    if queue_frame.empty:
        st.success("No emails are currently in the review queue for the active filters.")
        return

    queue_frame = queue_frame.sort_values(["spam_score", "id"], ascending=[False, False])
    preview = queue_frame[
        [
            "id",
            "date_label",
            "sender_domain",
            "sender_email",
            "subject",
            "spam_score",
            "spam_label",
            "processed_category",
            "review_status",
        ]
    ].rename(
        columns={
            "id": "Email ID",
            "date_label": "Date",
            "sender_domain": "Domain",
            "sender_email": "Sender",
            "subject": "Subject",
            "spam_score": "Score",
            "spam_label": "Predicted",
            "processed_category": "Cluster",
            "review_status": "Reviewed",
        }
    )
    st.dataframe(preview, use_container_width=True, hide_index=True)


def render_email_detail_tab(filtered_frame: pd.DataFrame) -> None:
    st.subheader("Email Detail")
    if filtered_frame.empty:
        st.info("No email rows match the active filters.")
        return

    email_ids = filtered_frame["id"].tolist()
    default_index = 0
    selected_id = st.selectbox("Select email", email_ids, index=default_index, format_func=lambda value: f"#{value}")
    row = filtered_frame.loc[filtered_frame["id"] == selected_id].iloc[0]

    cols = st.columns([1.3, 0.9])
    with cols[0]:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(f"### {row['subject']}")
        st.caption(f"{row['sender_email']} • {row['date_label']}")
        st.markdown(
            f"<span class='pill'>Score {format_score(row['spam_score'])}</span>"
            f"<span class='pill'>{row['spam_label']}</span>"
            f"<span class='pill'>{row['processed_category']}</span>",
            unsafe_allow_html=True,
        )
        st.write("**Spam reasons**")
        st.write(row["spam_reasons"] or "No explanation available.")
        st.write("**Authentication**")
        st.write(
            {
                "SPF": row["spf_result"],
                "DKIM": row["dkim_result"],
                "DMARC": row["dmarc_result"],
                "Content-Type": row["content_type"],
                "Mailer": row["mailer"],
            }
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with cols[1]:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.write("**Message metadata**")
        st.write(
            {
                "Email ID": int(row["id"]),
                "Message ID": row["message_id"],
                "Sender domain": row["sender_domain"],
                "Recipient": row["recipient_email"],
                "Processed flag": row["processed_flag"],
                "Reviewed label": row["ground_truth_label"] or "unreviewed",
            }
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Raw header JSON"):
        st.code(row["raw_headers_json"] or "{}", language="json")


def render_cluster_tab(filtered_frame: pd.DataFrame) -> None:
    st.subheader("Cluster Explorer")
    if filtered_frame.empty:
        st.info("No clustered emails match the current filters.")
        return

    cluster_summary = (
        filtered_frame.groupby("processed_category")
        .agg(
            email_count=("id", "count"),
            avg_score=("spam_score", "mean"),
            suspicious_count=("spam_label", lambda values: int(values.isin(["suspicious", "likely_spam"]).sum())),
            reviewed_count=("ground_truth_label", lambda values: int(values.notna().sum())),
        )
        .reset_index()
        .sort_values(["email_count", "avg_score"], ascending=[False, False])
    )
    st.dataframe(cluster_summary, use_container_width=True, hide_index=True)

    selected_cluster = st.selectbox(
        "Inspect cluster",
        cluster_summary["processed_category"].tolist(),
    )
    cluster_rows = filtered_frame[filtered_frame["processed_category"] == selected_cluster].copy()
    st.write(f"Top emails in `{selected_cluster}`")
    st.dataframe(
        cluster_rows[
            ["id", "date_label", "sender_domain", "subject", "spam_score", "spam_label", "review_status"]
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_review_tab(db_path: str, filtered_frame: pd.DataFrame) -> None:
    st.subheader("Review Lab")
    if filtered_frame.empty:
        st.info("No emails are available for review with the active filters.")
        return

    review_candidates = filtered_frame.sort_values(["spam_score", "id"], ascending=[False, False])
    selected_id = st.selectbox(
        "Pick an email to label",
        review_candidates["id"].tolist(),
        format_func=lambda value: f"#{value}",
        key="review_email_select",
    )
    row = review_candidates.loc[review_candidates["id"] == selected_id].iloc[0]

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.write(f"**Subject:** {row['subject']}")
    st.write(f"**Sender:** {row['sender_email']}")
    st.write(f"**Predicted label:** {row['spam_label']} ({format_score(row['spam_score'])})")
    st.write(f"**Reasons:** {row['spam_reasons']}")
    st.markdown("</div>", unsafe_allow_html=True)

    with st.form("review_form", clear_on_submit=False):
        default_index = LABEL_OPTIONS.index(row["ground_truth_label"]) if row["ground_truth_label"] in LABEL_OPTIONS else 0
        selected_label = st.selectbox("Ground-truth label", LABEL_OPTIONS, index=default_index)
        review_notes = st.text_area("Review notes", value=row["review_notes"] or "", height=110)
        reviewed_at = st.text_input("Reviewed at", value=row["reviewed_at"] or pd.Timestamp.utcnow().strftime("%Y-%m-%d"))
        submitted = st.form_submit_button("Save review label")

    if submitted:
        save_review_label(db_path, int(row["id"]), selected_label, review_notes, reviewed_at)
        st.success(f"Saved review for email #{int(row['id'])}. Refreshing dashboard data.")
        st.cache_data.clear()
        st.rerun()


def main() -> None:
    configure_page()

    db_path = st.sidebar.text_input("Database path", DEFAULT_DB_PATH)
    output_dir = st.sidebar.text_input("Artifact directory", DEFAULT_OUTPUT_DIR)

    email_frame = load_email_frame(db_path)
    summary = load_summary(output_dir)
    artifact_evaluation_summary = load_evaluation_summary(output_dir)
    evaluation_summary = compute_live_evaluation(email_frame) if not email_frame.empty else artifact_evaluation_summary

    render_header(summary, email_frame)

    if email_frame.empty:
        st.warning(
            "The database does not currently contain email rows. Run the ingestion pipeline and then the classifier before using the dashboard."
        )
        return

    filters = render_sidebar(email_frame)
    filtered_frame = apply_filters(email_frame, filters)

    overview_tab, queue_tab, detail_tab, cluster_tab, review_tab = st.tabs(
        ["Overview", "Review Queue", "Email Detail", "Cluster Explorer", "Review Lab"]
    )

    with overview_tab:
        render_overview_tab(summary, evaluation_summary, filtered_frame)
    with queue_tab:
        render_queue_tab(filtered_frame)
    with detail_tab:
        render_email_detail_tab(filtered_frame)
    with cluster_tab:
        render_cluster_tab(filtered_frame)
    with review_tab:
        render_review_tab(db_path, filtered_frame)


if __name__ == "__main__":
    main()
