"""
This module is designed to facilitate the running of all unit tests across the various submodules of the PCB Defect Detection system. It uses Python's built-in unittest framework to run tests located in the 'tests' subdirectory of each module.
"""

import os
import unittest

import source.utils.tests.test_runner as utils_tests
import source.image_preprocessing.tests.test_runner as image_preprocessing_tests
from source.utils import SimplePopupHandler
from source.utils import TestResultLogger


FILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(FILE_DIR, 'test_results.log')


def run_tests():
    test_suite = unittest.TestSuite()
    
    # Load tests from test modules
    test_suite = utils_tests.load_tests(test_suite)
    test_suite = image_preprocessing_tests.load_tests(test_suite)
    
    result = unittest.TextTestRunner(verbosity=2).run(test_suite)
    
    num_passed_tests = result.testsRun - len(result.errors) - len(result.failures) - len(result.skipped)
    runned_tests = result.testsRun - len(result.skipped)
    
    if num_passed_tests == runned_tests:
        message = f'All tests passed! ({num_passed_tests}/{runned_tests})'
    else:
        message = (f'{num_passed_tests} out of {runned_tests} tests passed.\n'
                   f'Failures: {len(result.failures)}\n'
                   f'Errors: {len(result.errors)}')
    
    return message

if __name__ == '__main__':

    os.makedirs(FILE_DIR, exist_ok=True)    
    TestResultLogger(LOG_FILE)

    message = run_tests()
    popup_handler = SimplePopupHandler()
    popup_handler.display_popup_message(message)
