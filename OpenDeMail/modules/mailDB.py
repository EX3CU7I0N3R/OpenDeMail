import json
import logging
import re
import sqlite3
from email.utils import parseaddr
from typing import Iterable


class MailDB:
    CREATE_TABLE_IF_NOT_EXISTS_SQL = """
    CREATE TABLE IF NOT EXISTS emails (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message_id TEXT NOT NULL UNIQUE,
        delivered_to TEXT,
        recipient_email TEXT,
        return_path TEXT,
        sender_name TEXT,
        sender_email TEXT,
        sender_domain TEXT,
        reply_to TEXT,
        subject TEXT,
        content_type TEXT,
        mime_version TEXT,
        content_transfer_encoding TEXT,
        mailer TEXT,
        date_header TEXT,
        inj_time INTEGER,
        fnc_id TEXT,
        spf_result TEXT,
        dkim_result TEXT,
        dmarc_result TEXT,
        arc_seal TEXT,
        arc_message_signature TEXT,
        arc_authentication_results TEXT,
        authentication_results TEXT,
        dkim_signature TEXT,
        received_spf TEXT,
        received TEXT,
        x_received TEXT,
        x_google_smtp_source TEXT,
        x_mailer TEXT,
        x_inj_time TEXT,
        x_fncid TEXT,
        raw_headers_json TEXT NOT NULL,
        processed_category TEXT,
        processed_flag TEXT,
        spam_score REAL,
        spam_label TEXT,
        spam_reasons TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """

    REQUIRED_COLUMNS = {
        "processed_category": "TEXT",
        "processed_flag": "TEXT",
        "spam_score": "REAL",
        "spam_label": "TEXT",
        "spam_reasons": "TEXT",
    }

    def __init__(self, db_name: str = "emails.db"):
        self.db_name = db_name
        self.conn = sqlite3.connect(self.db_name)
        self.create_table()

    def create_table(self) -> None:
        try:
            cursor = self.conn.cursor()
            cursor.execute(self.CREATE_TABLE_IF_NOT_EXISTS_SQL)
            self._ensure_required_columns(cursor)
            self.conn.commit()
            logging.info("Database table ready: %s", self.db_name)
        except sqlite3.Error as exc:
            logging.error("Error creating database table: %s", exc)
            raise

    def _ensure_required_columns(self, cursor: sqlite3.Cursor) -> None:
        existing_columns = {
            row[1]
            for row in cursor.execute("PRAGMA table_info(emails)").fetchall()
        }
        for column_name, column_type in self.REQUIRED_COLUMNS.items():
            if column_name not in existing_columns:
                cursor.execute(f"ALTER TABLE emails ADD COLUMN {column_name} {column_type}")
                logging.info("Added missing column '%s' to emails table", column_name)

    def close(self) -> None:
        try:
            self.conn.close()
        except sqlite3.Error as exc:
            logging.warning("Error closing database connection: %s", exc)

    def parse_auth_result(self, auth_results: str, auth_type: str) -> str:
        try:
            pattern = re.compile(rf"{auth_type}=([^\s;]+)")
            match = pattern.search(auth_results or "")
            if match:
                return match.group(1)
            return "none"
        except Exception as exc:
            logging.error("Error parsing %s result: %s", auth_type, exc)
            return "error"

    @staticmethod
    def _get_header(email_data: dict, key: str, default: str = "") -> str:
        value = email_data.get(key, default)
        if isinstance(value, list):
            return value[0] if value else default
        return value

    @staticmethod
    def _join_header_values(email_data: dict, key: str) -> str:
        value = email_data.get(key, "")
        if isinstance(value, list):
            return "\n".join(value)
        return value or ""

    @staticmethod
    def _safe_int(value) -> int:
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return 0

    def normalize_email_data(self, email_data: dict) -> dict:
        from_header = self._get_header(email_data, "From")
        sender_name, sender_email = parseaddr(from_header)
        sender_domain = sender_email.split("@")[-1] if "@" in sender_email else ""
        auth_results = self._get_header(email_data, "Authentication-Results")

        normalized_data = {
            "message_id": self._get_header(email_data, "Message-ID"),
            "delivered_to": self._get_header(email_data, "Delivered-To"),
            "recipient_email": self._get_header(email_data, "To"),
            "return_path": self._get_header(email_data, "Return-Path"),
            "sender_name": sender_name,
            "sender_email": sender_email,
            "sender_domain": sender_domain,
            "reply_to": self._get_header(email_data, "Reply-To"),
            "subject": self._get_header(email_data, "Subject"),
            "content_type": self._get_header(email_data, "Content-Type"),
            "mime_version": self._get_header(email_data, "MIME-Version"),
            "content_transfer_encoding": self._get_header(email_data, "Content-Transfer-Encoding"),
            "mailer": self._get_header(email_data, "X-Mailer"),
            "date_header": self._get_header(email_data, "Date"),
            "inj_time": self._safe_int(self._get_header(email_data, "X-Inj-Time")),
            "fnc_id": self._get_header(email_data, "X-Fncid"),
            "spf_result": self.parse_auth_result(auth_results, "spf"),
            "dkim_result": self.parse_auth_result(auth_results, "dkim"),
            "dmarc_result": self.parse_auth_result(auth_results, "dmarc"),
            "arc_seal": self._get_header(email_data, "ARC-Seal"),
            "arc_message_signature": self._get_header(email_data, "ARC-Message-Signature"),
            "arc_authentication_results": self._get_header(email_data, "ARC-Authentication-Results"),
            "authentication_results": auth_results,
            "dkim_signature": self._get_header(email_data, "DKIM-Signature"),
            "received_spf": self._get_header(email_data, "Received-SPF"),
            "received": self._join_header_values(email_data, "Received"),
            "x_received": self._join_header_values(email_data, "X-Received"),
            "x_google_smtp_source": self._get_header(email_data, "X-Google-Smtp-Source"),
            "x_mailer": self._get_header(email_data, "X-Mailer"),
            "x_inj_time": self._get_header(email_data, "X-Inj-Time"),
            "x_fncid": self._get_header(email_data, "X-Fncid"),
            "raw_headers_json": json.dumps(email_data, ensure_ascii=True, sort_keys=True),
            "processed_category": "",
            "processed_flag": "",
        }
        return normalized_data

    def insert_email(self, normalized_email_data: dict) -> bool:
        try:
            cursor = self.conn.cursor()
            columns = ", ".join(normalized_email_data.keys())
            placeholders = ", ".join("?" for _ in normalized_email_data)
            sql = f"INSERT INTO emails ({columns}) VALUES ({placeholders})"
            cursor.execute(sql, tuple(normalized_email_data.values()))
            self.conn.commit()
            logging.info("Inserted email %s", normalized_email_data.get("message_id", ""))
            return True
        except sqlite3.IntegrityError as exc:
            logging.info("Skipping duplicate email %s: %s", normalized_email_data.get("message_id", ""), exc)
            return False
        except sqlite3.Error as exc:
            logging.error("Error inserting email data: %s", exc)
            raise

    def show_all_emails(self) -> list[tuple]:
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM emails")
            return cursor.fetchall()
        except sqlite3.Error as exc:
            logging.error("Error fetching emails: %s", exc)
            raise

    def bulk_update_classification(self, updates: Iterable[tuple[int, str, str, float, str, str]]) -> None:
        try:
            cursor = self.conn.cursor()
            cursor.executemany(
                """
                UPDATE emails
                SET processed_category = ?, processed_flag = ?, spam_score = ?, spam_label = ?, spam_reasons = ?
                WHERE id = ?
                """,
                (
                    (
                        category,
                        flag,
                        spam_score,
                        spam_label,
                        spam_reasons,
                        email_id,
                    )
                    for email_id, category, flag, spam_score, spam_label, spam_reasons in updates
                ),
            )
            self.conn.commit()
            logging.info("Updated classification metadata for %s emails", cursor.rowcount)
        except sqlite3.Error as exc:
            logging.error("Error updating classification metadata: %s", exc)
            raise
