import os
import unittest

from source.utils.recursive_type_conversion import recursive_type_conversion
from source.utils.test_result_logger import TestResultLogger


ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..','..')
OUTPUT_DIR = os.path.join(ROOT_DIR, r'source/utils/tests/outputs')
LOG_FILE = os.path.join(OUTPUT_DIR, 'test_results.log')


class TestRecursiveTypeConversion(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.logger = TestResultLogger(LOG_FILE, 'Recursive Type Conversion Test')

    def tearDown(self):
        self.logger.log_test_outcome(self._outcome.result, self._testMethodName)

    def test_primitive_conversion(self):
        self.assertEqual(recursive_type_conversion("123", int), 123)
        self.assertEqual(recursive_type_conversion("123.456", float), 123.456)
        self.assertEqual(recursive_type_conversion("True", bool), True)
        self.assertEqual(recursive_type_conversion(123, str), "123")

    def test_list_conversion(self):
        self.assertEqual(recursive_type_conversion(["1", "2", "3"], [int, int, int]), [1, 2, 3])
        self.assertEqual(recursive_type_conversion(("1", 2, ""), [str, int, bool]), ['1', 2, False])

    def test_tuple_conversion(self):
        self.assertEqual(recursive_type_conversion(["1", "2", "True"], (int, int, bool)), (1, 2, True))
        self.assertEqual(recursive_type_conversion(("1", 2, "3"), (str, str, int)), ('1', '2', 3))

    def test_dict_conversion(self):
        self.assertEqual(recursive_type_conversion({"key1": "123", "key2": 456}, {"key1": int, "key2": str}), {"key1": 123, "key2": '456'})

    def test_dict_conversion_1(self):
        source = {
            "number_str": "123",
            "list_of_str": ["1", "2", "3"],
            "nested_dict": {
                "bool_str": "True"
            },
            "tuple_of_mixed": ('30','',['30',10])
        }
        template = {
            "number_str": int,
            "list_of_str": [int,int,int],
            "nested_dict": {
                "bool_str": bool
            },
            "tuple_of_mixed": (int,bool,[int, str])
        }
        expected = {
            "number_str": 123,
            "list_of_str": [1, 2, 3],
            "nested_dict": {
                "bool_str": True
            },
            "tuple_of_mixed": (30,False,[30,'10'])
        }
        self.assertEqual(recursive_type_conversion(source, template), expected)

    def test_dict_conversion_2(self):
        # Missing Key
        source = {
            "number_str": "123",
            "list_of_str": ["1", "2", "3"],
        }
        template = {
            "number_str": int,
            "list_of_str": [int,int,int],
            "tuple_of_mixed": (int,bool,[int, str])
        }
        expected = {
            "number_str": 123,
            "list_of_str": [1, 2, 3],
        }
        self.assertEqual(recursive_type_conversion(source, template), expected)

    def test_error_handling(self):
        with self.assertRaises(TypeError):
            recursive_type_conversion("not a list", [int])
        with self.assertRaises(TypeError):
            recursive_type_conversion("not a tuple", (int, str, bool))
        with self.assertRaises(TypeError):
            recursive_type_conversion(["not an int"], int)
        with self.assertRaises(TypeError):
            recursive_type_conversion([1,"string but missing bool"], (int, str, bool))
        with self.assertRaises(TypeError):
            recursive_type_conversion((1,"string but missing bool"), [int, str, bool])

if __name__ == '__main__':
    unittest.main()
