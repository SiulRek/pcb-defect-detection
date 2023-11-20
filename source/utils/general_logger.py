import logging
from logging.handlers import RotatingFileHandler
import os

class Logger:
    def __init__(self, log_file_path, log_level=logging.INFO, max_log_size=10*1024*1024, backup_count=5):
        """
        Initializes the Logger class with basic configuration.

        Args:
            log_file_path (str): Path to the log file.
            log_level (logging.level): Level of logging. Default is logging.INFO.
            max_log_size (int): Maximum size in bytes before rotating the log file. Default is 10MB.
            backup_count (int): Number of backup log files to keep. Default is 5.
        """
        self.log_file_path = log_file_path
        self.log_level = log_level
        self.max_log_size = max_log_size
        self.backup_count = backup_count

        # Ensure the directory for the log file exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        handler = RotatingFileHandler(log_file_path, maxBytes=max_log_size, backupCount=backup_count)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def info(self, message):
        """Writes an info message to the log."""
        self.logger.info(message)

    def warning(self, message):
        """Writes a warning message to the log."""
        self.logger.warning(message)

    def error(self, message):
        """Writes an error message to the log."""
        self.logger.error(message)


if __name__ == '__main__':
    # Example usage:
    logger = Logger(r'.\logfile.log')
    logger.info('This is an info message')
    logger.warning('This is a warning message')
    logger.error('This is an error message')
