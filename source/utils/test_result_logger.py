import unittest
import logging

from source.utils.general_logger import Logger

class TestResultLogger(Logger):
    """ A logger class for logging the results of unittest executions.

    This class is designed to log the outcome of tests run using Python's unittest framework. 
    It can log test passes, failures, and errors to a specified log file.

    This class inherits from the parent class `Logger`
    """
    
    def __init__(self, log_file='./test_results.log'):
        """
        Initialize the TestResultLogger with a specified log file.

        Args:
            log_file (str): The path to the log file. Defaults to 'test_results.log'.
        """
        super().__init__(log_file)

    def setup_logger(self):
        """
        Set up the logger with a file handler and a standard logging format.

        This method configures the logger to write to the log file specified in 
        the `log_file` attribute. If the logger already has handlers, it doesn't add new ones. 
        Additionally exception are printed but not raised to ensure the running of tests without 
        interruption. Because of the aforementioned features the parent child method is overwritten.
        """
        try:
            self.logger = logging.getLogger('TestResultLogger')
            if not self.logger.handlers:  # Check if the logger already has handlers
                self.logger.setLevel(logging.INFO)
                file_handler = logging.FileHandler(self.log_file, mode='w')
                file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                self.logger.addHandler(file_handler)
        except Exception as exc:
             print(f"Logging error: {exc}")
    
    def log_test_outcome(self, result, test_method_name):
        """ Log the outcome of a single test case.

        This method logs whether a specific test case passed, failed, or raised an error.

        Args:
            result (unittest.TestResult): The result object containing the outcomes of the test suite.
            test_method_name (str): The name of the test method.

        Usage Example in a Test Suite:
            Inside a unittest.TestCase subclass, call this method in the tearDown method to log the outcome of each test:
            
            class ExampleTest(unittest.TestCase):
                @classmethod
                def setUpClass(cls):
                    cls.logger = TestResultLogger()

                def tearDown(self):
                    self.logger.log_test_outcome(self._outcome.result, self._testMethodName)
        """
        try:
            success = True

            for error in result.errors:
                if error[0]._testMethodName == test_method_name:
                    success = False
                    message = 'Test raised exc: {}'.format(test_method_name)
                    if error[1] is not None:
                        message += '\n' + 'Message: {}'.format(error[1])
                    self.logger.error(message)
                    break


            for failure in result.failures:
                if failure[0]._testMethodName == test_method_name:
                    success = False
                    message = 'Test Failed: {}'.format(test_method_name)
                    if failure[1] is not None:
                        message += '\n' + 'Message: {}'.format(failure[1])
                    self.logger.error(message)
                    break

            if success: 
                self.logger.info('Test Passed: %s', test_method_name)
        except Exception as exc:
            print('Logging error: %s', exc)


# Example Test Suite to try out the logger
class MyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logger = TestResultLogger()
        cls.logger.log_title('servus')

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

if __name__ == '__main__':
    unittest.main()
