import os
import unittest
from unittest import defaultTestLoader as Loader

from source.utils import TestResultLogger
from source.utils.tests import get_sample_from_distribution_test

ROOT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."
)
OUTPUT_DIR = os.path.join(ROOT_DIR, r"source/utils/tests/outputs")
LOG_FILE = os.path.join(OUTPUT_DIR, "test_results.log")


def load_tests(test_suite):
    test_suite.addTest(Loader.loadTestsFromModule(get_sample_from_distribution_test))
    return test_suite


if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    TestResultLogger(LOG_FILE)  # Initialize Test Result Logger.

    test_suite = unittest.TestSuite()
    unittest.TextTestRunner().run(load_tests(test_suite))
