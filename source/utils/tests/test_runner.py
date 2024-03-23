import os
import unittest

from source.utils.tests import get_sample_from_distribution_test, image_classifiers_trainer_test, recursive_type_conversion_test
from source.utils.tests import class_instance_serializer_test, parse_and_repeat_test, copy_json_exclude_entries_test
from source.utils.tests import randomly_select_sequential_keys_test

from source.utils.test_result_logger import TestResultLogger


ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..','..')
OUTPUT_DIR = os.path.join(ROOT_DIR, r'source/utils/tests/outputs')
LOG_FILE = os.path.join(OUTPUT_DIR, 'test_results.log')


def load_tests(test_suite):
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromModule(get_sample_from_distribution_test))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromModule(recursive_type_conversion_test))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromModule(class_instance_serializer_test))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromModule(parse_and_repeat_test))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromModule(copy_json_exclude_entries_test))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromModule(randomly_select_sequential_keys_test))
    # test_suite.addTest(unittest.defaultTestLoader.loadTestsFromModule(image_classifiers_trainer_test))
    return test_suite

if __name__ == '__main__':

    TestResultLogger(LOG_FILE) # Initialize Test Result Logger.

    test_suite = unittest.TestSuite()
    unittest.TextTestRunner().run(load_tests(test_suite))