# simple entry point for the OpenDeMail application
from .modules.mailClient import MailClient
from .modules.mailParser import MailParser
import dotenv
import os
import logging

dotenv.load_dotenv()
username = os.getenv("EMAIL_USERNAME")
password = os.getenv("EMAIL_PASSWORD")
server = os.getenv("EMAIL_SERVER")
port = int(os.getenv("EMAIL_PORT"))
logging.info("Environment variables loaded successfully")

def main() -> None:
    LoginClient = MailClient(server=server, port=port).login(username, password)
    if LoginClient:
        logging.info("Login successful!")
        mail_parser = MailParser(LoginClient)
        latest_email = mail_parser.get_latest_email()
        total_emails = mail_parser.get_total_emails()
        logging.info(f"Total emails: {total_emails}")
        headers = mail_parser.fetch_headers(latest_email)
        logging.info(f"Latest email headers: {headers}")
        all_headers = mail_parser.fetch_all_headers()
        logging.info(f"All email headers: {all_headers}")
        filterbysender = mail_parser.filter_by_sender(all_headers, "alert@bigbasket.in")
        logging.info(f"Filter by sender result: {filterbysender}")
        
    LoginClient.logout()

if __name__ == "__main__":
    main()
