import unittest

from source.image_preprocessing.tests import step_base_test, multiple_steps_test, image_preprocessing_test

def load_tests(test_suite):
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromModule(step_base_test))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromModule(multiple_steps_test))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromModule(image_preprocessing_test))
    return test_suite

if __name__ == '__main__':
    test_suite = unittest.TestSuite()
    unittest.TextTestRunner().run(load_tests(test_suite))
