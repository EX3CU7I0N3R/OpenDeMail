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
        self.assertEqual(
            headers["Received"],
            ["by mx1.example.com", "by mx2.example.com"],
        )


class MailDBTests(unittest.TestCase):
    def test_normalize_and_insert_email(self):
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
            self.assertEqual(len(db.show_all_emails()), 1)
            db.close()


class ClassificationTests(unittest.TestCase):
    def test_spam_scoring_marks_auth_failures_and_keywords_as_spam(self):
        row = pd.Series(
            {
                "spf_pass": 0,
                "dkim_pass": 0,
                "dmarc_pass": 0,
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
            }
        )

        score, label, reasons = EmailClassifier._score_single_email(row)

        self.assertGreaterEqual(score, 70)
        self.assertEqual(label, "likely_spam")
        self.assertIn("SPF did not pass", reasons)
        self.assertIn("Promotional language detected", reasons)

    def test_spam_scoring_marks_clean_mail_as_ham(self):
        row = pd.Series(
            {
                "spf_pass": 1,
                "dkim_pass": 1,
                "dmarc_pass": 1,
                "promo_term_hits": 0,
                "phishing_term_hits": 0,
                "received_hops": 2,
                "exclamation_count": 0,
                "uppercase_ratio": 0.1,
                "subject_token_count": 4,
                "free_mailbox_sender": 0,
                "unknown_sender_domain": 0,
                "is_html": 1,
                "processed_category": "transactional_example_invoice",
            }
        )

        score, label, reasons = EmailClassifier._score_single_email(row)

        self.assertLess(score, 40)
        self.assertEqual(label, "likely_ham")
        self.assertEqual(reasons, "No strong spam indicators were triggered")


if __name__ == "__main__":
    unittest.main()
