# OpenDeMail

OpenDeMail fetches email headers over IMAP, stores normalized metadata in SQLite, clusters mailbox traffic into behavioral groups, scores spam risk, and now adds a manual-review and evaluation workflow so model quality can be measured over time.

## What The Project Does

- Connects to an IMAP mailbox using credentials from `.env`
- Pulls email headers from `INBOX`
- Normalizes sender, routing, authentication, and content metadata
- Stores records in `emails.db`
- Segments the mailbox with unsupervised clustering
- Scores each email for spam risk with trust-aware heuristics
- Imports manual review labels for evaluation
- Generates CSV, JSON, Markdown, and PNG outputs for analysis

## Project Structure

```text
OpenDeMail/
  __main__.py                Application entrypoint for mailbox ingestion
  classification.py          Clustering, scoring, evaluation, and artifact export
  modules/
    mailClient.py            IMAP connection lifecycle
    mailParser.py            Header extraction and parsing helpers
    mailDB.py                SQLite schema, inserts, review labels, and updates
tests/
  test_opendemail.py         Parser, DB, scoring, and evaluation tests
classification_output/       Generated reports, CSV exports, and charts
emails.db                    SQLite mailbox database
opendemail.log               Runtime log file
```

## Requirements

- Python 3.9+
- An IMAP-enabled mailbox
- Packages used by the project:
- `python-dotenv`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `scipy`
 - `streamlit`

Install dependencies:

```powershell
python -m pip install --upgrade pip setuptools
python -m pip install python-dotenv numpy pandas matplotlib scikit-learn scipy streamlit
```

## Environment Setup

Create `.env` in the project root:

```env
EMAIL_USERNAME="your-email@example.com"
EMAIL_PASSWORD="your-app-password"
EMAIL_SERVER="imap.gmail.com"
EMAIL_PORT=993
```

Notes:

- `EMAIL_PORT` must be an integer with no extra spaces around the value.
- Gmail usually requires an app password instead of your normal account password.

## Core Workflow

### 1. Ingest Mail Headers

Run the ingestion pipeline:

```powershell
python -m OpenDeMail
```

What happens:

1. Logging is configured.
2. Environment variables are loaded and validated.
3. IMAP login is established.
4. All message headers in `INBOX` are fetched.
5. Headers are normalized and inserted into `emails.db`.
6. Duplicate emails are skipped using `message_id`.

### 2. Run Classification, Scoring, And Evaluation

Run the analysis workflow:

```powershell
python -m OpenDeMail.classification
```

Useful options:

```powershell
python -m OpenDeMail.classification --incremental
python -m OpenDeMail.classification --start-date 2026-01-01 --end-date 2026-03-31
python -m OpenDeMail.classification --review-csv classification_output\manual_reviews.csv
```

What happens:

1. Email rows are loaded from `emails.db`, optionally filtered by date or incremental mode.
2. Feature engineering is applied to subject text, authentication results, content type, routing depth, and trust-history signals.
3. `KMeans` clusters similar emails into mailbox categories.
4. A deterministic spam scoring layer assigns:
   - `spam_score`
   - `spam_label`
   - `spam_reasons`
5. Classification fields are saved back to SQLite.
6. Optional manual review labels are imported into the `review_labels` table.
7. Predictions are compared against reviewed labels to generate evaluation metrics and false-positive summaries.
8. Reports, plots, exports, and review templates are written to `classification_output/`.

## Database Fields

### `emails` table

- `processed_category`: cluster/category name such as `trusted_icicibank.com_invoice`
- `processed_flag`: `normal` or `review`
- `spam_score`: 0-100 risk score
- `spam_label`: `likely_ham`, `suspicious`, or `likely_spam`
- `spam_reasons`: plain-English explanation of the triggered rules

### `review_labels` table

- `email_id`: foreign key to `emails.id`
- `ground_truth_label`: reviewed label
- `review_notes`: analyst note about the decision
- `reviewed_at`: optional review timestamp

`mailDB.py` creates missing columns and the manual review table automatically during startup.

## How Scoring Works Now

OpenDeMail still uses heuristic scoring, but the model is now more trust-aware.

Risk signals:

- SPF, DKIM, and DMARC failures
- Promotional and phishing-style subject terms
- High mail hop counts
- Multiple exclamation marks or uppercase-heavy subjects
- Unknown sender domains
- Cluster context from `unverified_*` or `marketing_*` groups

Trust signals:

- Sender frequency in the mailbox
- Stable sender-recipient history
- Recurring authenticated transactional domains
- Trusted personal mailbox patterns

This reduces false positives for recurring legitimate messages like transactional notifications and personal Gmail conversations.

## Manual Review Workflow

After each classification run, OpenDeMail writes:

- `classification_output/manual_review_candidates.csv`

This file contains the highest-risk emails plus blank columns for:

- `ground_truth_label`
- `review_notes`
- `reviewed_at`

Fill in labels using:

- `likely_ham`
- `suspicious`
- `likely_spam`

Then rerun:

```powershell
python -m OpenDeMail.classification --review-csv classification_output\manual_reviews.csv
```

The labels are imported into SQLite and used for evaluation.

## Generated Outputs

After running `python -m OpenDeMail.classification`, the `classification_output/` folder contains:

- `classification_summary.json`
  - Machine-readable summary of clusters, spam counts, filters, and evaluation data
- `classification_explanation.md`
  - Human-readable explanation of the pipeline and cluster behavior
- `evaluation_summary.json`
  - Metrics from reviewed labels, including precision, recall, and confusion counts
- `evaluation_report.md`
  - Human-readable review of false positives and false negatives
- `classified_emails.csv`
  - Row-level export including cluster, spam, and trust-signal fields
- `manual_review_candidates.csv`
  - Review queue template for adding ground-truth labels
- `cluster_distribution.png`
- `cluster_projection.png`
- `auth_rates_by_category.png`
- `top_sender_domains.png`
- `spam_label_distribution.png`
- `spam_score_histogram.png`
- `spam_by_category.png`

## Reading The Results

Use the outputs in this order:

1. Open `classification_summary.json` for the mailbox-wide summary.
2. Read `evaluation_summary.json` to see how reviewed labels compare to predictions.
3. Read `evaluation_report.md` for false positives and false negatives.
4. Read `classification_explanation.md` for the scoring and trust logic.
5. Check `spam_label_distribution.png` and `spam_score_histogram.png` to understand the overall spam-risk spread.
6. Open `classified_emails.csv` to inspect individual emails and their `spam_reasons`.

## Dashboard

Launch the dashboard with:

```powershell
streamlit run OpenDeMail\dashboard.py
```

The dashboard reads directly from `emails.db`, so it remains useful even if the latest artifact export came from an incremental no-op run.

Included views:

- Overview
  - mailbox totals, sender-domain breakdown, and latest evaluation snapshot
- Review Queue
  - triage table for suspicious and likely-spam emails
- Email Detail
  - message-level context, reasons, auth results, and raw headers
- Cluster Explorer
  - cluster-level counts, average scores, and constituent emails
- Review Lab
  - label emails as `likely_ham`, `suspicious`, or `likely_spam` and save review notes back to SQLite

## Development And Tests

Run tests:

```powershell
python -m unittest discover -s tests -v
```

Compile check:

```powershell
python -m compileall OpenDeMail tests
```

## Known Limits

- The spam classifier is still heuristic, not trained on labeled examples.
- The clustering output depends on mailbox contents and can change as new mail is inserted.
- Date filters rely on parseable `Date` headers.
- Analysis is based on headers and metadata, not full email bodies.

## Recommended Next Steps

- Add a small Streamlit dashboard for browsing the review queue and clusters.
- Add more reviewed labels to improve evaluation coverage.
- Track analysis runs historically so score drift and cluster drift can be compared over time.
- Consider a trained classifier only after enough reviewed examples exist.
