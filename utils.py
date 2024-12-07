import logging
import os
from logging.handlers import RotatingFileHandler
from collections import deque
from datetime import datetime
from config import LOG_FILE, LOG_QUEUE_MAX_SIZE, LOG_LEVEL

# Initialize the log queue with the size from config
LOG_QUEUE = deque(maxlen=LOG_QUEUE_MAX_SIZE)

def setup_logger():
    # Ensure the log directory exists
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    # Get the logger instance
    logger = logging.getLogger("simulation_app")

    # Check if the logger already has handlers
    if not logger.hasHandlers():
        # Set up the RotatingFileHandler
        rotating_handler = RotatingFileHandler(LOG_FILE, mode='a', maxBytes=10**6, backupCount=3)
        rotating_handler.setLevel(LOG_LEVEL)

        # Console output handler (optional)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(LOG_LEVEL)

        # Log formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        rotating_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(rotating_handler)
        logger.addHandler(console_handler)

        # Set the global logging level
        logger.setLevel(LOG_LEVEL)

    return logger

# Initialize logger once at the start
logger = setup_logger()

def add_log(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"

    # Append to the global log queue
    LOG_QUEUE.append(formatted_message)

    # Log the message
    logger.info(formatted_message)

    # Ensure logs are flushed to the file
    for handler in logger.handlers:
        if isinstance(handler, RotatingFileHandler):
            handler.flush()

# Function to reinitialize the log queue with a new size
def adjust_log_queue_size(new_maxlen: int):
    global LOG_QUEUE
    current_logs = list(LOG_QUEUE)  # Copy current logs
    LOG_QUEUE.clear()  # Clear the old queue
    LOG_QUEUE = deque(current_logs, maxlen=new_maxlen)  # Set new size
