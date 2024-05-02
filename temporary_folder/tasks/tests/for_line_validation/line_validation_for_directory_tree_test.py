import unittest

from temporary_folder.tasks.constants.defaults import DIRECTORY_TREE_DEFAULTS as DEFAULTS
from temporary_folder.tasks.helpers.for_create_query.line_validation import line_validation_for_directory_tree


class TestLineValidationForDirectoryTree(unittest.TestCase):
    def test_directory_tree_with_defaults(self):
        line = "#tree example"
        result = line_validation_for_directory_tree(line)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "example")
        self.assertEqual(result[1], DEFAULTS.MAX_DEPTH.value)
        self.assertEqual(result[2], DEFAULTS.INCLUDE_FILES.value)
        self.assertEqual(result[3], DEFAULTS.IGNORE_LIST.value)

    def test_directory_tree_with_full_arguments(self):
        line = "#tree example (5, true, [temp; log])"
        result = line_validation_for_directory_tree(line)
        self.assertEqual(result[1], 5)
        self.assertTrue(result[2])
        self.assertEqual(result[3], ['temp', 'log'])


    def test_directory_tree_with_partial_arguments(self):
        line = "#tree example (5, true)"
        result = line_validation_for_directory_tree(line)
        self.assertEqual(result[1], 5)
        self.assertTrue(result[2])
        self.assertEqual(result[3], DEFAULTS.IGNORE_LIST.value)
    
    def test_directory_tree_with_one_argument(self):
        line = "#tree example (5)"
        result = line_validation_for_directory_tree(line)
        self.assertEqual(result[1], 5)
        self.assertEqual(result[2], DEFAULTS.INCLUDE_FILES.value)
        self.assertEqual(result[3], DEFAULTS.IGNORE_LIST.value)


if __name__ == '__main__':
    unittest.main()
