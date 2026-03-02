import logging
from .modules.mailClient import MailClient
from .modules.mailParser import MailParser

# Set up logging
# Create logfile if it doesn't exist and log to it
logging.basicConfig(

    filename='opendemail.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')


logging.info("OpenDeMail application started")