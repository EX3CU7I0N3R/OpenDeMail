import OpenDeMail.modules.mailClient as mailClient
import logging
import email
import re

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
    

    # fetch all email headers from inbox and return as a list of dictionaries
    def fetch_headers(self, raw_email: bytes) -> dict:
        try:
            email_message = email.message_from_bytes(raw_email)
            headers = dict(email_message.items())
            return headers
        except Exception as e:
            logging.error(f"Error fetching email headers: {e}")
            return {}
        
    #fetch all email headers from inbox and return as a list of dictionaries
    def fetch_all_headers(self) -> list:
        m = self.mail_client
        m.select("INBOX")
        try:
            result, data = m.search(None, "ALL")
            mail_ids = data[0].split()
            if not mail_ids:
                return []

            # fetch headers for all messages in a single command
            msg_set = b",".join(mail_ids)
            result, data = m.fetch(msg_set, "(RFC822.HEADER)")
            headers_list = []
            for resp in data:
                if isinstance(resp, tuple):
                    msg = email.message_from_bytes(resp[1])
                    headers_list.append(dict(msg.items()))
            return headers_list

        except Exception as e:
            logging.error(f"Error fetching all email headers: {e}")
            return []
    
    #Apply regex to "From" key where the sender email is store between "<" and ">" and check if it matches the sender_email in a nested dictionary of headers and return the list of records that match the sender email
    def filter_by_sender(self, headers: dict, sender_email: str) -> bool:
        try:
            filtered_headers = {}
            sender_pattern = re.compile(r"<([^>]+)>")
            for header in headers:
                from_field = header.get("From", "")
                match = sender_pattern.search(from_field)
                if match and match.group(1) == sender_email:
                    # return True
                    filtered_headers[header.get("Subject", "No Subject")] = header
            return filtered_headers if filtered_headers else False
            # return False
        except Exception as e:
            logging.error(f"Error filtering by sender: {e}")
            return False