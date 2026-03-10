import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

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


if __name__ == "__main__":
    unittest.main()
