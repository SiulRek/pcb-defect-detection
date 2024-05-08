import unittest
from temporary_folder.tasks.helpers.for_cleanup.split_import_statement import get_alias_names


class TestGetAliasNames(unittest.TestCase):
    def test_single_name(self):
        self.assertEqual(get_alias_names('ceil'), ['ceil'], "Failed on single name without aliases or commas")

    def test_multiple_names_commas(self):
        self.assertEqual(get_alias_names('np, dt, sqrt'), ['np', 'dt', 'sqrt'], "Failed on list of names separated by commas without spaces")

    def test_multiple_names_with_spaces_and_commas(self):
        self.assertEqual(get_alias_names('np  , dt ,  sqrt'), ['np', 'dt', 'sqrt'], "Failed on list of names separated by commas with spaces")

    def test_parentheses_with_newlines(self):
        self.assertEqual(get_alias_names('(\nsqrt,\nceil,\nfloor\n)'), ['sqrt', 'ceil', 'floor'], "Failed on names within parentheses with newlines")

    def test_empty_string(self):
        self.assertEqual(get_alias_names(''), [], "Failed on empty string input")


if __name__ == '__main__':
    unittest.main()
