import logging
import os

from constants.constants import LOGS_FOLDER


def set_competition_hash_folder(output_folder):
    global COMPETITION_HASH_FOLDER
    COMPETITION_HASH_FOLDER = output_folder

def setup_logger(name, log_file, level=logging.INFO) -> logging.Logger:
    """
    Function to set up a logger; it can log to both file and console.

    :param name: Name of the logger.
    :param log_file: Path to the log file.
    :param level: Logging level.
    :return: Custom logger.
    """

    # Create a custom logger
    logger = logging.getLogger(name)

    # Path to the log folder
    log_folder = os.path.join(COMPETITION_HASH_FOLDER, LOGS_FOLDER)

    # Path to the logger file
    log_file = os.path.join(log_folder, log_file)

    # Create a log folder if it does not exist
    os.makedirs(os.path.join(COMPETITION_HASH_FOLDER, os.path.dirname(log_file)), exist_ok=True)

    # Check if the logger has handlers, to prevent adding duplicate handlers
    if not logger.hasHandlers():
        # Set the logging level
        logger.setLevel(level)

        # Create handlers
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(log_file)

        # Define format to include time, folder, file name, line number, and message
        formatter = logging.Formatter(
            '%(asctime)s - %(pathname)s - %(filename)s - %(lineno)d - %(levelname)s - %(message)s'
        )

        # Set the formatter for both console and file handlers
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
