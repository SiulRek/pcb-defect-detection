import unittest
from source.utils import parse_and_repeat

class TestParseAndRepeat(unittest.TestCase):

    def test_basic_functionality(self):
        self.assertEqual(parse_and_repeat('[1]*3 + [4]*2 + [True] + ["String"]'), [1, 1, 1, 4, 4, True, "String"])

    def test_nested_lists(self):
        self.assertEqual(parse_and_repeat('[[1,2,"3.22"]]*2 + [[3,True,1.0]]*3 + [[4,False,2.0]]'),
                         [[1, 2, '3.22'], [1, 2, '3.22'], [3, True, 1.0], [3, True, 1.0], [3, True, 1.0],[4,False,2.0]])

    def test_no_repetition(self):
        self.assertEqual(parse_and_repeat('[5] + [6] + ["test"]'), [5, 6, "test"])

    def test_empty_list(self):
        self.assertEqual(parse_and_repeat('[]'), [])

    def test_single_element(self):
        self.assertEqual(parse_and_repeat('[7]'), [7])
        self.assertEqual(parse_and_repeat('[7]*1'), [7])

    def test_string_representation(self):
        self.assertEqual(parse_and_repeat('[""] + ["World", "True"]*2 + [["World", "True"]]*2'),
                         ['',"World", "True","World", "True",["World", "True"], ["World", "True"]])
    
    def test_dict_representation(self):
        self.assertEqual(parse_and_repeat('[{"key": 3}]*2 + [{"key": 1}]'),
                         [{"key": 3},{"key": 3},{"key": 1}])

    def test_invalid_formatting(self):
        with self.assertRaises(ValueError):
            parse_and_repeat('[1, 2, 3]*a')
        with self.assertRaises(ValueError):
            parse_and_repeat('')
        with self.assertRaises(ValueError):
            parse_and_repeat('+[3]')
        with self.assertRaises(ValueError):
            parse_and_repeat('[3]+')
        with self.assertRaises(ValueError):
            parse_and_repeat('3*6')
        with self.assertRaises(ValueError):
            parse_and_repeat('[3] +, [10]')
        
    def test_invalid_literal(self):
        with self.assertRaises(ValueError):
            parse_and_repeat('[3] + 10]')
        with self.assertRaises(ValueError):
            parse_and_repeat('[[1,2,"3.22"]]*2 + [[3,True,Invalid]]*3 ')
        with self.assertRaises(ValueError):
            parse_and_repeat('[[1,2+3]]*2 + [[3,True]]*3 ')
        with self.assertRaises(ValueError):
            parse_and_repeat('[[1,2,"3.22"]]*2 + [[3,True,,]]*3 ')


if __name__ == '__main__':
    unittest.main()
