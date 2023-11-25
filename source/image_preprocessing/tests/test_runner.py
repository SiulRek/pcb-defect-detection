import unittest

from source.image_preprocessing.tests import step_base_test, multiple_steps_test, image_preprocessing_test
from source.image_preprocessing.tests.channel_conversions_steps_test import load_channel_conversion_tests


def load_tests(test_suite):
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromModule(step_base_test))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromModule(multiple_steps_test))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromModule(image_preprocessing_test))
    test_suite.addTest(load_channel_conversion_tests())
    return test_suite

if __name__ == '__main__':
    test_suite = unittest.TestSuite()
    unittest.TextTestRunner().run(load_tests(test_suite))
