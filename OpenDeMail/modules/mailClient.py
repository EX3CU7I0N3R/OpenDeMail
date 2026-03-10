import imaplib
import logging
from typing import Optional


class MailClient:
    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        port: Optional[int] = None,
        timeout: int = 15,
    ) -> None:
        self.username = username
        self.password = password
        self.server = server
        self.port = port
        self.timeout = timeout
        self.connection: Optional[imaplib.IMAP4_SSL] = None

    def login(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> Optional[imaplib.IMAP4_SSL]:
        self.username = username or self.username
        self.password = password or self.password

        try:
            self.connection = imaplib.IMAP4_SSL(self.server, self.port, timeout=self.timeout)
            self.connection.login(self.username, self.password)
            logging.info("Connected to %s:%s as %s", self.server, self.port, self.username)
            return self.connection
        except (imaplib.IMAP4.error, OSError) as exc:
            logging.error("IMAP login failed: %s", exc)
            self.connection = None
            return None

    def logout(self) -> None:
        if self.connection is None:
            return

        try:
            self.connection.logout()
            logging.info("Logged out of %s:%s", self.server, self.port)
        except imaplib.IMAP4.error as exc:
            logging.warning("IMAP logout failed: %s", exc)
        finally:
            self.connection = None
