import os
import logging

def setup_logger(log_file=None, log_level=logging.INFO):
    # Create or get the logger
    logger = logging.getLogger(__name__)
    
    # Set log level
    logger.setLevel(log_level)
    
