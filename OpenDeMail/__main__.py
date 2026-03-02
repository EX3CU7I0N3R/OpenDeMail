# simple entry point for the OpenDeMail application
from modules.mailClient import MailClient
from modules.mailParser import MailParser
import dotenv
import os

dotenv.load_dotenv()
username = os.getenv("EMAIL_USERNAME")
password = os.getenv("EMAIL_PASSWORD")
def main() -> None:
    LoginClient = MailClient().login(username, password)
    if LoginClient:
        print("Login successful!")
        mail_parser = MailParser(LoginClient)
        latest_email = mail_parser.get_latest_email()
        print(latest_email)

if __name__ == "__main__":
    main()
