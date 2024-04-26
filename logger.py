# logger.py
import logging
import os

def setup_logger(log_file=None, log_level=logging.INFO):
    """
    Set up the logger with the specified log file and log level.

    Args:
        log_file (str, optional): The path to the log file. If not provided, logs will be written to the console.
        log_level (int, optional): The log level. Defaults to logging.INFO.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger('fracksec')
    logger.setLevel(log_level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger