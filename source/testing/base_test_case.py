from abc import ABC
import os
import unittest

from source.testing.helpers.load_dataset_from_tf_records import (
    load_dataset_from_tf_records,
)
from source.testing.helpers.test_result_logger import TestResultLogger

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image_data")


class BaseTestCase(unittest.TestCase, ABC):
    """
    Abstract base class for all test cases, providing setup and teardown
    operations that are common across various tests. This class ensures that all
    derived test classes are prepared with necessary directories and logging
    capabilities.

    Attributes:
        - output_dir (str): Directory where test outputs will be saved.
        - temp_dir (str): Temporary directory for use during tests.
        - log_file (str): Path to the log file used to record test results.
        - logger (TestResultLogger): Logger instance to log test outcomes.
    """

    @classmethod
    def compute_output_dir(cls, parent_folder="tests"):
        """
        Computes the directory path for test outputs by traversing up the file
        hierarchy until a directory named 'tests' is found. It then returns the
        path to the 'outputs' subdirectory within 'tests'.

        Args:
            - parent_folder (str, optional): The name of the parent folder
                containing the 'tests' directory.

        Returns:
            - str: The path to the output directory.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        while parent_folder not in os.listdir(current_dir):
            current_dir = os.path.dirname(current_dir)
            if os.path.splitdrive(current_dir)[1] == os.sep:
                msg = "Tests directory not found in the path hierarchy."
                raise NotADirectoryError(msg)

        return os.path.join(current_dir, parent_folder, "outputs")

    @classmethod
    def get_test_case_name(cls):
        """
        Generates a more readable test case name by removing 'Test' prefix and
        adding spaces before each capital letter in the class name.

        Returns:
            - str: The formatted test case name.
        """
        name = cls.__name__
        if name.startswith("Test"):
            name = name[4:]
        name = [letter if letter.islower() else f" {letter}" for letter in name]
        return "".join(name)

    @classmethod
    def setUpClass(cls):
        """ Class-level setup method that ensures necessary directories are created
        and initializes logging for the test case. """
        cls.output_dir = cls.compute_output_dir()
        if not os.path.exists(cls.output_dir):
            os.makedirs(cls.output_dir)
        cls.temp_dir = os.path.join(cls.output_dir, "temp")
        if not os.path.exists(cls.temp_dir):
            os.makedirs(cls.temp_dir)

        cls.log_file = os.path.join(cls.output_dir, "test_results.log")
        cls.logger = TestResultLogger(cls.log_file, cls.get_test_case_name())

    @classmethod
    def tearDownClass(cls):
        """ Class-level teardown method that removes the temporary directory created
        during the test setup. """
        if os.path.exists(cls.temp_dir):
            os.rmdir(cls.temp_dir)

    def tearDown(self):
        """ Instance-level teardown method that logs the outcome of each test
        method. """
        self.logger.log_test_outcome(self._outcome.result, self._testMethodName)

    def load_image_dataset(self):
        """
        Load the image dataset used for testing. This method is intended to be
        overridden by derived test classes to return the appropriate dataset.

        Returns:
            - tf.data.Dataset: The image dataset to be used for testing.
        """
        tf_records_path = os.path.join(
            DATA_DIR, "tf_records", "geometrical_forms.tfrecord"
        )
        return load_dataset_from_tf_records(tf_records_path)
