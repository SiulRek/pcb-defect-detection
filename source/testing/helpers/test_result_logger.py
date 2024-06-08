import logging
import unittest


class TestResultLogger:
    """
    A Singleton logger class for logging the results of unittest executions.
    There is an optional setup_logger step to initialize file handlers and
    logging format for results. This class generates two types of logs: detailed
    and simplified version.

    Detailed Log:
        - Includes specific information about the status of each test -
            whether it passed, failed, or raised an exception.
        - Simplified Log: Only includes whether the test passed or not.
    """

    _instance = None
    _setup_file_handlers = True
    _failures_count = 0
    _errors_count = 0

    def __new__(cls, *args, **kwargs):
        """
        Ensures a single instance of the TestResultLogger class. If an instance
        already exists, it returns that instance, else it creates a new one.

        Returns:
            - TestResultLogger: instance of TestResultLogger class
        """
        if cls._instance is None:
            cls._instance = super(TestResultLogger, cls).__new__(cls)
        else:
            cls._setup_file_handlers = False
        return cls._instance

    def __init__(self, log_file="./test_results.log", title=""):
        """
        Initialize the TestResultLogger with specified log file and optional
        title. If this is the first instance, the logger setup is called unless
        the _skip_setup variable is True. Optionally, a title can be logged at
        the creation of the instance.

        Args:
            - log_file (str): The path to the detailed log file. Defaults to
                'test_results.log'.
            - title (str): The title to be logged when TestResultLogger is
                initialized. Defaults to '' (no title).
        """
        self.log_file = log_file
        self.log_file_simple = log_file.replace(".log", "_simple.log")
        self.setup_logger()
        if title:
            self.log_title(title)

    def _setup_file_handler(self, logger, file_path):
        """
        Setup a file handler for the logger.

        Args:
            - logger (logging.Logger): Logger object to configure.
            - file_path (str): File path for logging.
        """
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(file_path, mode="w")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

    def setup_logger(self):
        """
        Set up the loggers with file handlers and a standard logging format.

        This method configures two loggers: one for detailed test outcomes and
        another for simplified outcomes. Each logger writes to its respective
        log file.
        """
        # Logger for detailed test outcomes
        self.logger = logging.getLogger("TestResultLogger")
        # Logger for simplified test outcomes
        self.simple_logger = logging.getLogger("TestResultLoggerSimple")

        if self._setup_file_handlers:
            self._setup_file_handler(self.logger, self.log_file)
            self._setup_file_handler(self.simple_logger, self.log_file_simple)

    def log_title(self, title):
        """
        Logs a title string with a specific format.

        This method formats the given title string with a pattern (dashes before
        and after) and logs it at the INFO level.

        Args:
            - title (str): Title to be logged.
        """
        formatted_title = "-" * 14 + f" {title} " + "-" * (60 - len(title) - 14)
        self.logger.info(formatted_title)
        self.simple_logger.info(formatted_title)

    def _log_outcome(self, outcome_type, test_method_name, message=""):
        """
        Logs a detailed outcome (raised exc or failure) including the message.

        Args:
            - outcome_type (str): The type of outcome ('passed', 'raised
                exc' or 'failure').
            - test_method_name (str): The name of the test method.
            - message (str): The error or failure message.
        """
        log_message = f"Test {outcome_type}: {test_method_name}"
        if outcome_type == "passed":
            self.simple_logger.info(log_message)
            self.logger.info(log_message)
        elif outcome_type in ["failure", "raised exc"]:
            self.simple_logger.error(log_message)
            log_message += f"\nMessage: {message}"
            self.logger.error(log_message)
        else:
            raise ValueError(
                f"Outcome Type: '{outcome_type}' is not recognized from Logger."
            )

    def log_test_outcome(self, result, test_method_name):
        """
        Log the outcome of a single test case.

        Logs detailed outcomes to one file and simplified outcomes (pass/fail)
        to another.

        Args:
            - result (unittest.TestResult): The result object containing the
                outcomes of the test suite.
            - test_method_name (str): The name of the test method.
        """
        try:
            success = True

            if self._errors_count + 1 == len(result.errors):
                success = False
                error = result.errors[self._errors_count]
                self._log_outcome("raised exc", test_method_name, error[1])
                self._errors_count += 1

            if self._failures_count + 1 == len(result.failures):
                success = False
                failure = result.failures[self._failures_count]
                self._log_outcome("failure", test_method_name, failure[1])
                self._failures_count += 1

            if success:
                self._log_outcome("passed", test_method_name)
        except Exception as exc:  # The Logger shall not interrupt the testing process.
            print("Logging error: %s", exc)


# Example Test Suite to try out the logger
class MyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logger = TestResultLogger()
        cls.logger.log_title("This is a Title")

    def tearDown(self):
        # Use the logger to log the outcome of each test
        self.logger.log_test_outcome(self._outcome.result, self._testMethodName)

    @classmethod
    def tearDownClass(cls) -> None:
        return super().tearDownClass()

    def test_example_pass(self):
        self.assertEqual(1, 1)

    def test_example_fail(self):
        self.assertEqual(1, 2)

    def test_example_error(self):
        raise ValueError()


if __name__ == "__main__":
    unittest.main()
