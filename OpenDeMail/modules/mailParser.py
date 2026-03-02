import OpenDeMail.modules.mailClient as mailClient

class MailParser:
    def __init__(self, mail_client: mailClient.MailClient):
        self.mail_client = mail_client

    #get lastest email from the inbox
    def get_latest_email(self):
        mail = self.mail_client
        mail.select("inbox")
        result, data = mail.search(None, "ALL")
        email_ids = data[0].split()
        latest_email_id = email_ids[-1]
        result, data = mail.fetch(latest_email_id, "(RFC822)")
        raw_email = data[0][1]
        return raw_email