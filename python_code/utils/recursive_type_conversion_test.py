import unittest

from python_code.utils.recursive_type_conversion import recursive_type_conversion


class TestRecursiveTypeConversion(unittest.TestCase):

    def test_primitive_conversion(self):
        self.assertEqual(recursive_type_conversion("123", 1), 123)
        self.assertEqual(recursive_type_conversion("123.456", 1.0), 123.456)
        self.assertEqual(recursive_type_conversion("True", True), True)
        self.assertEqual(recursive_type_conversion(123, "string"), "123")

    def test_list_conversion(self):
        self.assertEqual(recursive_type_conversion(["1", "2", "3"], [1, 2, 3]), [1, 2, 3])
        self.assertEqual(recursive_type_conversion(("1", 2, ""), [1, '3', True]), [1, '2', False])

    def test_tuple_conversion(self):
        self.assertEqual(recursive_type_conversion(["1", "2", "True"], (1, 2, False)), (1, 2, True))
        self.assertEqual(recursive_type_conversion(("1", 2, "3"), (1, '2', 3)), (1, '2', 3))

    def test_dict_conversion(self):
        self.assertEqual(recursive_type_conversion({"key1": "123", "key2": 456}, {"key1": 1, "key2": '2'}), {"key1": 123, "key2": '456'})

    def test_recursive_conversion(self):
        source = {
            "number_str": "123",
            "list_of_str": ["1", "2", "3"],
            "nested_dict": {
                "bool_str": "True"
            }
        }
        template = {
            "number_str": 0,
            "list_of_str": [0, 0, 0],
            "nested_dict": {
                "bool_str": False
            }
        }
        expected = {
            "number_str": 123,
            "list_of_str": [1, 2, 3],
            "nested_dict": {
                "bool_str": True
            }
        }
        self.assertEqual(recursive_type_conversion(source, template), expected)

    def test_error_handling(self):
        with self.assertRaises(TypeError):
            recursive_type_conversion("not a list", [1, 2, 3])
        with self.assertRaises(TypeError):
            recursive_type_conversion("not a tuple", (1, 2, 3))

if __name__ == '__main__':
    unittest.main()