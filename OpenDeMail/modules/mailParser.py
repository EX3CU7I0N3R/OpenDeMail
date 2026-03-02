import OpenDeMail.modules.mailClient as mailClient
import logging


class MailParser:
    def __init__(self, mail_client: mailClient.MailClient):
        self.mail_client = mail_client

    #get lastest email from the inbox
    def get_latest_email(self)-> bytes:
        mail = self.mail_client
        mail.select("INBOX")
        try:
            result, data = mail.search(None, "ALL")
            mail_ids = data[0].split()
            latest_email_id = mail_ids[-1]
            result, data = mail.fetch(latest_email_id, "(RFC822)")
            raw_email = data[0][1]
        except Exception as e:
            logging.error(f"Error searching emails: {e}")
            return b""
        return raw_email
    
    def get_total_emails(self)-> int:
        mail = self.mail_client
        mail.select("INBOX")
        try:
            result, data = mail.search(None, "ALL")
            mail_ids = data[0].split()
            return len(mail_ids)
        except Exception as e:
            logging.error(f"Error getting total emails: {e}")
            return 0