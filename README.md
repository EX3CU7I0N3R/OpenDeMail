# OpenDeMail

OpenDeMail fetches email headers over IMAP, stores normalized metadata in SQLite, clusters mailbox traffic into behavioral groups, and adds a spam-risk layer on top of those clusters.

## What The Project Does

- Connects to an IMAP mailbox using credentials from `.env`
- Pulls email headers from `INBOX`
- Normalizes sender, routing, authentication, and content metadata
- Stores records in `emails.db`
- Segments the mailbox with unsupervised clustering
- Scores each email for spam risk and records the reasons behind the score
- Generates CSV, JSON, Markdown, and PNG outputs for analysis

## Project Structure

```text
OpenDeMail/
  __main__.py                Application entrypoint for mailbox ingestion
  classification.py          Clustering + spam scoring workflow
  modules/
    mailClient.py            IMAP connection lifecycle
    mailParser.py            Header extraction and parsing helpers
    mailDB.py                SQLite schema, inserts, and updates
tests/
  test_opendemail.py         Parser, DB, and spam scoring tests
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

Install dependencies:

```powershell
python -m pip install --upgrade pip setuptools
python -m pip install python-dotenv numpy pandas matplotlib scikit-learn scipy
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

### 2. Run Classification And Spam Detection

Run the analysis workflow:

```powershell
python -m OpenDeMail.classification
```

What happens:

1. All rows are loaded from `emails.db`.
2. Feature engineering is applied to subject text, authentication results, content type, and routing depth.
3. `KMeans` clusters similar emails into mailbox categories.
4. A deterministic spam scoring layer assigns:
   - `spam_score`
   - `spam_label`
   - `spam_reasons`
5. The database is updated with:
   - `processed_category`
   - `processed_flag`
   - `spam_score`
   - `spam_label`
   - `spam_reasons`
6. Reports and plots are written to `classification_output/`.

## Database Fields Added By Analysis

The `emails` table stores ingestion data plus analysis fields:

- `processed_category`: cluster/category name such as `transactional_linkedin.com_updates`
- `processed_flag`: `normal` or `review`
- `spam_score`: 0-100 risk score
- `spam_label`: `likely_ham`, `suspicious`, or `likely_spam`
- `spam_reasons`: plain-English explanation of the triggered rules

If an existing `emails.db` is missing these columns, `mailDB.py` adds them automatically during startup.

## How Clustering Works

Clustering is unsupervised. The system does not use labeled spam/ham examples.

Features used for clustering include:

- Subject text TF-IDF vectors
- Sender domain and sender email context
- Mailer and content type
- SPF, DKIM, and DMARC results
- Received hop counts
- Promotional and phishing keyword counts
- Punctuation and uppercase patterns

Cluster names are generated from:

- Dominant sender domain
- Common subject keywords
- Overall trust pattern in that cluster

Examples:

- `transactional_icicibank.com_bank`
- `marketing_example.com_offer`
- `unverified_unknown_verify`

## How Spam Detection Works

Spam detection sits on top of clustering and uses weighted heuristics.

Strong signals:

- SPF failure or missing SPF
- DKIM failure or missing DKIM
- DMARC failure or missing DMARC

Medium signals:

- Promotional language in the subject
- Security, payment, login, or password keywords
- High mail hop count
- Multiple exclamation marks
- Uppercase-heavy subject lines
- Missing sender domain

Context signals:

- Free mailbox sender combined with promotional language
- HTML email with promotional subject
- Membership in an `unverified_*` or `marketing_*` cluster

Spam label thresholds:

- `likely_ham`: score below 40
- `suspicious`: score from 40 to below 70
- `likely_spam`: score 70 or above

This is a deterministic risk model. It is intended for ranking and triage, not as a guaranteed spam verdict.

## Generated Outputs

After running `python -m OpenDeMail.classification`, the `classification_output/` folder contains:

- `classification_summary.json`
  - Machine-readable summary of cluster sizes, spam counts, and top suspicious examples
- `classification_explanation.md`
  - Human-readable explanation of the pipeline, decisions, and graph meanings
- `classified_emails.csv`
  - Row-level export including cluster and spam fields
- `cluster_distribution.png`
  - Volume by predicted cluster
- `cluster_projection.png`
  - 2D projection of cluster placement
- `auth_rates_by_category.png`
  - SPF, DKIM, and DMARC pass ratios by cluster
- `top_sender_domains.png`
  - Top sender domains by message count
- `spam_label_distribution.png`
  - Counts of ham, suspicious, and spam labels
- `spam_score_histogram.png`
  - Distribution of spam scores
- `spam_by_category.png`
  - Stacked spam labels inside each predicted cluster

## Reading The Results

Use the outputs in this order:

1. Open `classification_summary.json` for the mailbox-wide summary.
2. Read `classification_explanation.md` for the decision logic.
3. Check `spam_label_distribution.png` and `spam_score_histogram.png` to understand the overall spam-risk spread.
4. Check `spam_by_category.png` to see which clusters deserve review first.
5. Open `classified_emails.csv` to inspect individual high-risk emails and their `spam_reasons`.

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

- The spam classifier is heuristic, not trained on labeled examples.
- The clustering output depends on the mailbox contents and can change as new mail is inserted.
- The pipeline currently reads all rows in the database at once.
- Analysis is based on headers and metadata, not full email bodies.

## Recommended Next Steps

- Add a small dashboard to browse high-risk emails interactively.
- Add date-window filtering for large mailboxes.
- Add model evaluation once manually labeled spam/ham examples exist.
- Store historical analysis runs so cluster drift can be tracked over time.
