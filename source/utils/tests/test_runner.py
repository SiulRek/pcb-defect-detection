import os
import unittest

from source.utils.tests import get_sample_from_distribution_test, recursive_type_conversion_test, class_instance_serializer_test, parse_and_repeat_test
from source.utils.test_result_logger import TestResultLogger


ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..','..')
OUTPUT_DIR = os.path.join(ROOT_DIR, r'source/utils/tests/outputs')
LOG_FILE = os.path.join(OUTPUT_DIR, 'test_result.log')


def load_tests(test_suite):
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromModule(get_sample_from_distribution_test))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromModule(recursive_type_conversion_test))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromModule(class_instance_serializer_test))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromModule(parse_and_repeat_test))
    return test_suite

if __name__ == '__main__':

    TestResultLogger(LOG_FILE) # Initialize Test Result Logger.

    test_suite = unittest.TestSuite()
    unittest.TextTestRunner().run(load_tests(test_suite))