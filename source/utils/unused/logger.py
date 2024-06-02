import logging
import os


class Logger:
    def __init__(self, log_file, log_level=logging.INFO):
        """
        Initializes the Logger class with basic configuration.

        Args:
            - log_file_path (str): Path to the log file.
            - log_level (logging.level): Level of logging. Default is
                logging.INFO.
        """
        self.log_file = log_file
        self.log_level = log_level
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self.setup_logger()

    def setup_logger(self):
        """
        Set up the logger with a file handler and a standard logging format.

        This method configures the logger to write to the log file specified in
        the `log_file` attribute.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.log_level)
        handler = logging.FileHandler(self.log_file, mode="w")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def close_logger(self):
        """ Close and remove all handlers attached to the logger. """
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

    def info(self, message):
        """ Writes an info message to the log. """
        self.logger.info(message)

    def warning(self, message):
        """ Writes a warning message to the log. """
        self.logger.warning(message)

    def error(self, message):
        """ Writes an error message to the log. """
        self.logger.error(message)

    def log_title(self, title):
        """
        Logs a title string with a specific format.

        This method formats the given title string with a pattern (dashes before
        and after) and logs it at the INFO level.
        """
        formatted_title = "-" * 14 + f" {title} " + "-" * (60 - len(title) - 14)
        self.logger.info(formatted_title)


if __name__ == "__main__":
    # Example usage:
    logger = Logger(r".\logfile.log")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
