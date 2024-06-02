import os
import unittest

from source.model.tests import image_classifiers_trainer_test
from source.utils import TestResultLogger

ROOT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."
)
OUTPUT_DIR = os.path.join(ROOT_DIR, r"source/model/tests/outputs")
LOG_FILE = os.path.join(OUTPUT_DIR, "test_results.log")


def load_tests(test_suite):
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromModule(image_classifiers_trainer_test)
    )
    return test_suite


if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    TestResultLogger(LOG_FILE)  # Initialize Test Result Logger.

    test_suite = unittest.TestSuite()
    unittest.TextTestRunner().run(load_tests(test_suite))
