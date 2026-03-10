import logging
import os

import dotenv

from .modules.mailClient import MailClient
from .modules.mailDB import MailDB
from .modules.mailParser import MailParser


def configure_logging() -> None:
    logging.basicConfig(
        filename="opendemail.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_mail_config() -> tuple[str, str, str, int]:
    dotenv.load_dotenv()

    username = (os.getenv("EMAIL_USERNAME") or "").strip()
    password = os.getenv("EMAIL_PASSWORD") or ""
    server = (os.getenv("EMAIL_SERVER") or "").strip()
    port_value = (os.getenv("EMAIL_PORT") or "").strip()

    missing = [
        name
        for name, value in (
            ("EMAIL_USERNAME", username),
            ("EMAIL_PASSWORD", password),
            ("EMAIL_SERVER", server),
            ("EMAIL_PORT", port_value),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    try:
        port = int(port_value)
    except ValueError as exc:
        raise ValueError("EMAIL_PORT must be a valid integer") from exc

    return username, password, server, port


def main() -> int:
    configure_logging()
    logging.info("OpenDeMail application started")

    mail_client = None
    mail_db = None

    try:
        username, password, server, port = load_mail_config()
        logging.info("Environment variables loaded successfully")

        mail_client = MailClient(server=server, port=port, username=username)
        connection = mail_client.login(username, password)
        if not connection:
            logging.error("Unable to establish IMAP session")
            return 1

        mail_parser = MailParser(connection)
        all_headers = mail_parser.fetch_all_headers()
        logging.info("Fetched %s email header records", len(all_headers))

        mail_db = MailDB()
        inserted_count = 0
        for header in all_headers:
            normalized_data = mail_db.normalize_email_data(header)
            if mail_db.insert_email(normalized_data):
                inserted_count += 1

        logging.info("Inserted %s new emails into the database", inserted_count)
        print(f"Fetched {len(all_headers)} emails and inserted {inserted_count} new records.")
        return 0
    except Exception as exc:
        logging.exception("Application failed: %s", exc)
        print(f"OpenDeMail failed: {exc}")
        return 1
    finally:
        if mail_db is not None:
            mail_db.close()
        if mail_client is not None:
            mail_client.logout()


if __name__ == "__main__":
    raise SystemExit(main())
