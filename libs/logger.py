# //////////////////// logger ////////////////////

#shows only WARNING, ERROR and CRITICAL and not INFO or DEBUG
#DEBUG do NOT goes in the logging.log file
import logging
import os

def logger():
    
    log_file = os.path.join("__pycache__", "logging.log")
    logging.basicConfig(filename=log_file, filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    logger.addHandler(console_handler)

# //////////////////// logger ////////////////////