import os
import unittest

from source.preprocessing.helpers.for_preprocessor.recursive_type_conversion import (
    recursive_type_conversion,
)
from source.testing.base_test_case import BaseTestCase


class TestRecursiveTypeConversion(BaseTestCase):
    """
    Unit tests for `recursive_type_conversion`.

    This suite tests the functionality of converting types within nested data
    structures based on a provided template. It covers a variety of data types
    including primitive types, lists, tuples, and dictionaries, ensuring that
    the function can handle complex nested transformations.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.test_data_directory = os.path.join(
            cls.output_dir, "recursive_type_conversion_tests"
        )
        os.makedirs(cls.test_data_directory, exist_ok=True)

    def test_primitive_conversion(self):
        self.assertEqual(recursive_type_conversion("123", int), 123)
        self.assertEqual(recursive_type_conversion("123.456", float), 123.456)
        self.assertEqual(recursive_type_conversion("True", bool), True)
        self.assertEqual(recursive_type_conversion(123, str), "123")

    def test_list_conversion(self):
        self.assertEqual(
            recursive_type_conversion(["1", "2", "3"], [int, int, int]), [1, 2, 3]
        )
        self.assertEqual(
            recursive_type_conversion(("1", 2, ""), [str, int, bool]), ["1", 2, False]
        )

    def test_tuple_conversion(self):
        self.assertEqual(
            recursive_type_conversion(["1", "2", "True"], (int, int, bool)),
            (1, 2, True),
        )
        self.assertEqual(
            recursive_type_conversion(("1", 2, "3"), (str, str, int)), ("1", "2", 3)
        )

    def test_dict_conversion(self):
        self.assertEqual(
            recursive_type_conversion(
                {"key1": "123", "key2": 456}, {"key1": int, "key2": str}
            ),
            {"key1": 123, "key2": "456"},
        )

    def test_dict_conversion_1(self):
        source = {
            "number_str": "123",
            "list_of_str": ["1", "2", "3"],
            "nested_dict": {"bool_str": "True"},
            "tuple_of_mixed": ("30", "", ["30", 10]),
        }
        template = {
            "number_str": int,
            "list_of_str": [int, int, int],
            "nested_dict": {"bool_str": bool},
            "tuple_of_mixed": (int, bool, [int, str]),
        }
        expected = {
            "number_str": 123,
            "list_of_str": [1, 2, 3],
            "nested_dict": {"bool_str": True},
            "tuple_of_mixed": (30, False, [30, "10"]),
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
            recursive_type_conversion([1, "string but missing bool"], (int, str, bool))
        with self.assertRaises(TypeError):
            recursive_type_conversion((1, "string but missing bool"), [int, str, bool])


if __name__ == "__main__":
    unittest.main()
