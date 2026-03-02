# Mail Class to represent an email message

import imaplib
import logging

class MailClient:
    def __init__(self, username: str = None, password: str = None, server: str = None, port: int = None) -> None:
        self.username = username
        self.password = password
        self.server = server
        self.port = port
    
    #login with imaplib
    def login(self, username: str, password: str) -> bool:
        try:
            mail = imaplib.IMAP4_SSL(self.server, self.port)
            mail.login(username, password)
            return mail
        except imaplib.IMAP4.error as e:
            logging.error(f"Error logging in: {e}")
            return False

    def logout(self, mail):
        mail.logout()
        print("Logged out successfully")
        logging.info(f"Logged out of {self.server}:{self.port} -> {self.username}")
    
