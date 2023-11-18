"""
This module is designed to facilitate the running of all unit tests across the various submodules of the PCB Defect Detection system. It uses Python's built-in unittest framework to run tests located in the 'tests' subdirectory of each module.
"""

import unittest

test_pattern = "test*.py"

def run_all_tests():
    """
    Loads and runs all unit tests in the specified submodules of the PCB Defect Detection system.

    Returns:
        TestResult: An instance of unittest.TestResult containing information about the test run.
    """
import unittest

import python_code.utils.tests.test_runner as utils_tests
import python_code.image_preprocessing.tests.test_runner as image_preprocessing_tests
from python_code.utils.simple_popup_handler import SimplePopupHandler

def run_tests():
    test_suite = unittest.TestSuite()
    
    # Load tests from test modules
    test_suite = utils_tests.load_tests(test_suite)
    test_suite = image_preprocessing_tests.load_tests(test_suite)
    
    result = unittest.TextTestRunner(verbosity=2).run(test_suite)
    
    num_passed_tests = result.testsRun - len(result.errors) - len(result.failures)
    
    if num_passed_tests == result.testsRun:
        message = f'All tests passed! ({num_passed_tests}/{result.testsRun})'
    else:
        message = (f'{num_passed_tests} out of {result.testsRun} tests passed.\n'
                   f'Failures: {len(result.failures)}\n'
                   f'Errors: {len(result.errors)}')
    
    return message

if __name__ == '__main__':
    message = run_tests()
    popup_handler = SimplePopupHandler()
    popup_handler.display_popup_message(message)
