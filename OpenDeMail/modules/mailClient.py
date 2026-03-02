# Mail Class to represent an email message

import imaplib

class MailClient:
    def __init__(self, username: str = None, password: str = None) -> None:
        self.username = username
        self.password = password
    
    #login with imaplib
    def login(self, username: str, password: str) -> bool:
        try:
            mail = imaplib.IMAP4_SSL("imap.gmail.com")
            mail.login(username, password)
            return mail
        except imaplib.IMAP4.error as e:
            print(f"Error logging in: {e}")
            return False
    