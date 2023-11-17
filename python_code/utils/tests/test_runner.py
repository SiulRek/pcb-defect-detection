import unittest

from python_code.utils.tests import get_sample_from_distribution_test, recursive_type_conversion_test, class_instance_serializer_test, parse_and_repeat_test

def load_tests():
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromModule(get_sample_from_distribution_test))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromModule(recursive_type_conversion_test))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromModule(class_instance_serializer_test))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromModule(parse_and_repeat_test))
    return test_suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(load_tests())
