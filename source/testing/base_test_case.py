from abc import ABC
import os
import shutil
import sys
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
    def _get_class_file_path(cls):
        module_name = cls.__module__
        module = sys.modules[module_name]
        return getattr(module, "__file__", "Module has no file location")

    @classmethod
    def _compute_output_dir(cls, parent_folder="tests"):
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
        current_dir = os.path.dirname(cls._get_class_file_path())
        while parent_folder not in os.listdir(current_dir):
            current_dir = os.path.dirname(current_dir)
            if current_dir == os.path.dirname(current_dir):
                msg = "Tests directory not found in the path hierarchy."
                raise NotADirectoryError(msg)

        return os.path.join(current_dir, parent_folder, "outputs")

    @classmethod
    def _get_test_case_name(cls):
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
        return "".join(name) + " Test"

    @classmethod
    def setUpClass(cls):
        """ Class-level setup method that ensures necessary directories are created
        and initializes logging for the test case. """
        cls.output_dir = cls._compute_output_dir()
        cls.temp_dir = os.path.join(cls.output_dir, "temp")

        os.makedirs(cls.output_dir, exist_ok=True)

        cls.log_file = os.path.join(cls.output_dir, "test_results.log")
        cls.logger = TestResultLogger(cls.log_file, cls._get_test_case_name())

    def setUp(self):
        """ Instance-level setup method that creates a temporary directory for use
        during the test. """
        os.makedirs(self.temp_dir, exist_ok=True)

    def tearDown(self):
        """ Instance-level teardown method that logs the outcome of each test method
        and removes the temporary directory created during the test setup. """
        self.logger.log_test_outcome(self._outcome.result, self._testMethodName)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @classmethod
    def load_image_dataset(cls):
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
