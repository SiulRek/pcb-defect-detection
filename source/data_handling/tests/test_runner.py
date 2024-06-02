import os
import unittest
from unittest import defaultTestLoader as Loader

from source.data_handling.tests import label_manager_test
from source.data_handling.tests import load_dataset_test
from source.data_handling.tests import split_dataset_test
from source.data_handling.tests import enhance_dataset_test
from source.data_handling.tests import tfrecord_serialization_test
from source.utils import TestResultLogger

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
OUTPUT_DIR = os.path.join(ROOT_DIR, r"source/data_handling/tests/outputs")
LOG_FILE = os.path.join(OUTPUT_DIR, "test_results.log")


def load_tests(test_suite):
    test_suite.addTest(Loader.loadTestsFromModule(load_dataset_test))
    test_suite.addTest(Loader.loadTestsFromModule(label_manager_test))
    test_suite.addTest(Loader.loadTestsFromModule(split_dataset_test))
    test_suite.addTest(Loader.loadTestsFromModule(enhance_dataset_test))
    test_suite.addTest(Loader.loadTestsFromModule(tfrecord_serialization_test))
    return test_suite


if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    TestResultLogger(LOG_FILE)  # Initialize Test Result Logger.

    test_suite = unittest.TestSuite()
    unittest.TextTestRunner().run(load_tests(test_suite))
