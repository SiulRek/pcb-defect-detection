import unittest

from source.image_preprocessing.tests import image_preprocessing_test, step_base_test
from source.image_preprocessing.tests.channel_conversions_steps_test import load_channel_conversion_steps_tests
from source.image_preprocessing.tests.multiple_steps_test import load_multiple_steps_tests
from source.image_preprocessing.tests.resize_operations_steps_test import load_resize_operations_steps_tests
from source.image_preprocessing.tests.long_pipeline_test import load_long_pipeline_tests


def load_tests(test_suite):
    """
    Populates the given test suite with a series of test cases from the image preprocessing testing framework.

    This function aggregates tests from different aspects of the image preprocessing pipeline. It includes basic tests,
    tests for multiple steps, specific channel conversion tests, and general image preprocessing tests. 

    Args:
        test_suite (unittest.TestSuite): The test suite to which the tests will be added.

    Returns:
        unittest.TestSuite: The test suite populated with a range of tests from different modules.
    """
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromModule(image_preprocessing_test))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromModule(step_base_test))
    test_suite.addTest(load_multiple_steps_tests())
    test_suite.addTest(load_channel_conversion_steps_tests())
    test_suite.addTest(load_resize_operations_steps_tests())
    test_suite.addTest(load_long_pipeline_tests(1)) 
    return test_suite


if __name__ == '__main__':
    """ Main execution block for running the aggregated test suite."""
    test_suite = unittest.TestSuite()
    unittest.TextTestRunner().run(load_tests(test_suite))
