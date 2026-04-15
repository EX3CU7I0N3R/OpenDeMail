import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from OpenDeMail.classification import EmailClassifier
from OpenDeMail.modules.mailDB import MailDB
from OpenDeMail.modules.mailParser import MailParser


class MailParserTests(unittest.TestCase):
    def test_fetch_headers_keeps_duplicate_received_headers(self):
        parser = MailParser(mail_client=None)
        raw_email = (
            b"From: Demo <demo@example.com>\r\n"
            b"Subject: Test\r\n"
            b"Received: by mx1.example.com\r\n"
            b"Received: by mx2.example.com\r\n"
            b"\r\n"
            b"Body"
        )

        headers = parser.fetch_headers(raw_email)

        self.assertEqual(headers["Subject"], "Test")
        self.assertEqual(headers["Received"], ["by mx1.example.com", "by mx2.example.com"])


class MailDBTests(unittest.TestCase):
    def test_normalize_insert_and_upsert_manual_reviews(self):
        with TemporaryDirectory() as temp_dir:
            db = MailDB(str(Path(temp_dir) / "emails.db"))
            email_data = {
                "Message-ID": "<1@example.com>",
                "From": "Sender <sender@example.com>",
                "To": "receiver@example.com",
                "Subject": "Hello",
                "Authentication-Results": "mx.example.com; spf=pass dkim=pass dmarc=pass",
                "Received": ["hop-one", "hop-two"],
                "X-Inj-Time": "123",
            }

            normalized = db.normalize_email_data(email_data)

            self.assertEqual(normalized["sender_email"], "sender@example.com")
            self.assertEqual(normalized["sender_domain"], "example.com")
            self.assertEqual(normalized["inj_time"], 123)
            self.assertTrue(db.insert_email(normalized))
            self.assertFalse(db.insert_email(normalized))

            db.bulk_upsert_manual_reviews([(1, "likely_ham", "known sender", "2026-04-13")])
            db.bulk_upsert_manual_reviews([(1, "suspicious", "updated review", "2026-04-14")])

            review_row = db.conn.execute(
                "SELECT ground_truth_label, review_notes, reviewed_at FROM review_labels WHERE email_id = 1"
            ).fetchone()
            self.assertEqual(review_row[0], "suspicious")
            self.assertEqual(review_row[1], "updated review")
            self.assertEqual(review_row[2], "2026-04-14")
            db.close()


class ClassificationTests(unittest.TestCase):
    def test_spam_scoring_marks_auth_failures_and_keywords_as_spam(self):
        row = pd.Series(
            {
                "spf_pass": 0,
                "dkim_pass": 0,
                "dmarc_pass": 0,
                "auth_failure_count": 3,
                "promo_term_hits": 2,
                "phishing_term_hits": 2,
                "received_hops": 5,
                "exclamation_count": 3,
                "uppercase_ratio": 0.6,
                "subject_token_count": 5,
                "free_mailbox_sender": 1,
                "unknown_sender_domain": 0,
                "is_html": 1,
                "processed_category": "unverified_unknown_offer",
                "trusted_sender_profile": 0,
                "trusted_free_mailbox_sender": 0,
                "recurring_transactional_domain": 0,
                "prior_interaction": 0,
                "recipient_specific_sender": 0,
            }
        )

        score, label, reasons = EmailClassifier._score_single_email(row)

        self.assertGreaterEqual(score, 70)
        self.assertEqual(label, "likely_spam")
        self.assertIn("SPF did not pass", reasons)
        self.assertIn("Promotional language detected", reasons)

    def test_spam_scoring_reduces_risk_for_trusted_transactional_mail(self):
        row = pd.Series(
            {
                "spf_pass": 0,
                "dkim_pass": 1,
                "dmarc_pass": 1,
                "auth_failure_count": 1,
                "promo_term_hits": 0,
                "phishing_term_hits": 1,
                "received_hops": 2,
                "exclamation_count": 0,
                "uppercase_ratio": 0.1,
                "subject_token_count": 4,
                "free_mailbox_sender": 0,
                "unknown_sender_domain": 0,
                "is_html": 1,
                "processed_category": "trusted_bank.example_invoice",
                "trusted_sender_profile": 1,
                "trusted_free_mailbox_sender": 0,
                "recurring_transactional_domain": 1,
                "prior_interaction": 1,
                "recipient_specific_sender": 1,
            }
        )

        score, label, reasons = EmailClassifier._score_single_email(row)

        self.assertLess(score, 40)
        self.assertEqual(label, "likely_ham")
        self.assertIn("trusted sender context", reasons)
        self.assertIn("lowers risk", reasons)

    def test_evaluate_predictions_reports_metrics_and_false_positives(self):
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "emails.db"
            classifier = EmailClassifier(db_path=str(db_path), output_dir=str(Path(temp_dir) / "out"))
            db = MailDB(str(db_path))
            db.bulk_upsert_manual_reviews(
                [
                    (1, "likely_ham", "personal message", "2026-04-13"),
                    (2, "likely_spam", "obvious phish", "2026-04-13"),
                    (3, "suspicious", "needs review", "2026-04-13"),
                ]
            )
            db.close()

            classified_frame = pd.DataFrame(
                [
                    {
                        "id": 1,
                        "sender_domain": "gmail.com",
                        "subject": "Fees payment proof",
                        "spam_score": 52.0,
                        "spam_label": "suspicious",
                        "spam_reasons": "Keyword hit",
                    },
                    {
                        "id": 2,
                        "sender_domain": "bad.example",
                        "subject": "Verify account now",
                        "spam_score": 90.0,
                        "spam_label": "likely_spam",
                        "spam_reasons": "Multiple failures",
                    },
                    {
                        "id": 3,
                        "sender_domain": "bank.example",
                        "subject": "Login alert",
                        "spam_score": 15.0,
                        "spam_label": "likely_ham",
                        "spam_reasons": "Trusted sender",
                    },
                ]
            )

            evaluation = classifier.evaluate_predictions(classified_frame)

            self.assertEqual(evaluation["reviewed_count"], 3)
            self.assertAlmostEqual(evaluation["exact_match_accuracy"], 1 / 3, places=4)
            self.assertAlmostEqual(evaluation["review_precision"], 0.5, places=4)
            self.assertAlmostEqual(evaluation["review_recall"], 0.5, places=4)
            self.assertEqual(len(evaluation["top_false_positives"]), 1)
            self.assertEqual(evaluation["top_false_positives"][0]["id"], 1)
            self.assertEqual(len(evaluation["top_false_negatives"]), 1)
            self.assertEqual(evaluation["top_false_negatives"][0]["id"], 3)

    def test_import_review_csv_persists_labels(self):
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "emails.db"
            review_csv = Path(temp_dir) / "reviews.csv"
            with review_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["email_id", "ground_truth_label", "review_notes", "reviewed_at"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "email_id": 17,
                        "ground_truth_label": "likely_ham",
                        "review_notes": "safe sender",
                        "reviewed_at": "2026-04-13",
                    }
                )

            classifier = EmailClassifier(
                db_path=str(db_path),
                output_dir=str(Path(temp_dir) / "out"),
                review_csv_path=str(review_csv),
            )

            imported_count = classifier.import_review_csv()

            self.assertEqual(imported_count, 1)
            db = MailDB(str(db_path))
            stored = db.conn.execute(
                "SELECT ground_truth_label, review_notes, reviewed_at FROM review_labels WHERE email_id = 17"
            ).fetchone()
            db.close()
            self.assertEqual(stored, ("likely_ham", "safe sender", "2026-04-13"))

    def test_incremental_run_with_no_unclassified_emails_returns_noop_summary(self):
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "emails.db"
            output_dir = Path(temp_dir) / "out"
            db = MailDB(str(db_path))
            db.insert_email(
                {
                    "message_id": "<processed@example.com>",
                    "delivered_to": "",
                    "recipient_email": "receiver@example.com",
                    "return_path": "",
                    "sender_name": "Sender",
                    "sender_email": "sender@example.com",
                    "sender_domain": "example.com",
                    "reply_to": "",
                    "subject": "Already processed",
                    "content_type": "text/plain",
                    "mime_version": "1.0",
                    "content_transfer_encoding": "",
                    "mailer": "mailer",
                    "date_header": "Mon, 13 Apr 2026 10:00:00 +0000",
                    "inj_time": 0,
                    "fnc_id": "",
                    "spf_result": "pass",
                    "dkim_result": "pass",
                    "dmarc_result": "pass",
                    "arc_seal": "",
                    "arc_message_signature": "",
                    "arc_authentication_results": "",
                    "authentication_results": "",
                    "dkim_signature": "",
                    "received_spf": "",
                    "received": "hop-one",
                    "x_received": "",
                    "x_google_smtp_source": "",
                    "x_mailer": "",
                    "x_inj_time": "",
                    "x_fncid": "",
                    "raw_headers_json": "{}",
                    "processed_category": "trusted_example.com_update",
                    "processed_flag": "normal",
                    "spam_score": 5.0,
                    "spam_label": "likely_ham",
                    "spam_reasons": "No strong spam indicators were triggered",
                }
            )
            db.close()

            classifier = EmailClassifier(
                db_path=str(db_path),
                output_dir=str(output_dir),
                incremental=True,
            )

            summary = classifier.run()

            self.assertEqual(summary["status"], "no_new_emails")
            self.assertTrue((output_dir / "classification_summary.json").exists())
            self.assertTrue((output_dir / "evaluation_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
