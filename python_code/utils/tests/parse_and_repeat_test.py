import unittest
from python_code.utils import parse_and_repeat

class TestParseAndRepeat(unittest.TestCase):

    def test_basic_functionality(self):
        self.assertEqual(parse_and_repeat('[1]*3, [4]*2, [True]'), [1, 1, 1, 4, 4, True])

    def test_nested_lists(self):
        self.assertEqual(parse_and_repeat('[[1,2,"3.22"]]*2,[[3,1,1.0]]*3'),
                         [[1, 2, '3.22'], [1, 2, '3.22'], [3, 1, 1.0], [3, 1, 1.0], [3, 1, 1.0]])

    def test_no_repetition(self):
        self.assertEqual(parse_and_repeat('[5], [6], ["test"]'), [5, 6, "test"])

    def test_empty_list(self):
        self.assertEqual(parse_and_repeat('[]'), [])

    def test_empty_string(self):
        self.assertEqual(parse_and_repeat(''), [])

    def test_single_element(self):
        self.assertEqual(parse_and_repeat('[7]*1'), [7])

    def test_string_representation(self):
        self.assertEqual(parse_and_repeat('["[8,9]", "[10,11]"]*2'),
                         ['[8,9]', '[10,11]', '[8,9]', '[10,11]'])

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            parse_and_repeat('[1, 2, 3]*a')

# Run the tests
if __name__ == '__main__':
    unittest.main()
