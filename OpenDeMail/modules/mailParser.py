import email
import logging
import re
from email import policy
from typing import Any


class MailParser:
    def __init__(self, mail_client) -> None:
        self.mail_client = mail_client

    @staticmethod
    def _message_headers_to_dict(message) -> dict[str, Any]:
        headers: dict[str, Any] = {}
        for key, value in message.raw_items():
            if key in headers:
                existing = headers[key]
                if isinstance(existing, list):
                    existing.append(value)
                else:
                    headers[key] = [existing, value]
            else:
                headers[key] = value
        return headers

    def get_latest_email(self) -> bytes:
        mail = self.mail_client
        mail.select("INBOX")
        try:
            _, data = mail.search(None, "ALL")
            mail_ids = data[0].split()
            if not mail_ids:
                return b""
            latest_email_id = mail_ids[-1]
            _, data = mail.fetch(latest_email_id, "(RFC822)")
            return data[0][1]
        except Exception as exc:
            logging.error("Error fetching latest email: %s", exc)
            return b""

    def get_total_emails(self) -> int:
        mail = self.mail_client
        mail.select("INBOX")
        try:
            _, data = mail.search(None, "ALL")
            mail_ids = data[0].split()
            return len(mail_ids)
        except Exception as exc:
            logging.error("Error getting total emails: %s", exc)
            return 0

    def fetch_headers(self, raw_email: bytes) -> dict:
        try:
            email_message = email.message_from_bytes(raw_email, policy=policy.default)
            return self._message_headers_to_dict(email_message)
        except Exception as exc:
            logging.error("Error fetching email headers: %s", exc)
            return {}

    def fetch_all_headers(self) -> list[dict]:
        mail = self.mail_client
        mail.select("INBOX")
        try:
            _, data = mail.search(None, "ALL")
            mail_ids = data[0].split()
            if not mail_ids:
                return []

            msg_set = b",".join(mail_ids)
            _, data = mail.fetch(msg_set, "(RFC822.HEADER)")
            headers_list = []
            for response in data:
                if isinstance(response, tuple):
                    msg = email.message_from_bytes(response[1], policy=policy.default)
                    headers_list.append(self._message_headers_to_dict(msg))
            return headers_list
        except Exception as exc:
            logging.error("Error fetching all email headers: %s", exc)
            return []

    def filter_by_sender(self, headers: list[dict], sender_email: str):
        try:
            filtered_headers = {}
            sender_pattern = re.compile(r"<([^>]+)>")
            for header in headers:
                from_field = header.get("From", "")
                if isinstance(from_field, list):
                    from_field = from_field[0]
                match = sender_pattern.search(from_field)
                if match and match.group(1).lower() == sender_email.lower():
                    filtered_headers[header.get("Subject", "No Subject")] = header
            return filtered_headers if filtered_headers else False
        except Exception as exc:
            logging.error("Error filtering by sender: %s", exc)
            return False
