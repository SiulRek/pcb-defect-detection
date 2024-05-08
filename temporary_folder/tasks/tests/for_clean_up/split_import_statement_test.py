import unittest

from temporary_folder.tasks.helpers.for_cleanup.split_import_statement import split_import_statement


class TestSplitImportStatement(unittest.TestCase):
    def test_simple_import(self):
        base, specifiers = split_import_statement('import os')
        self.assertEqual(base, 'import')
        self.assertEqual(specifiers, ['os'])

    def test_import_with_alias(self):
        base, specifier = split_import_statement('import numpy as np')
        self.assertEqual(base, 'import numpy as')
        self.assertEqual(specifier, ['np'])

    def test_from_import_single(self):
        base, specifier = split_import_statement('from sys import path')
        self.assertEqual(base, 'from sys import')
        self.assertEqual(specifier, ['path'])

    def test_from_import_multiple(self):
        base, specifier = split_import_statement('from datetime import datetime, timedelta')
        self.assertEqual(base, 'from datetime import')
        self.assertEqual(specifier, ['datetime', 'timedelta'])

    def test_from_import_with_parentheses(self):
        base, specifier = split_import_statement('from math import (\nsqrt,\nceil,\nfloor\n)')
        self.assertEqual(base, 'from math import')
        self.assertEqual(specifier, ['sqrt', 'ceil', 'floor'])

    def test_from_import_with_alias(self):
        base, specifier = split_import_statement('from pandas.core import series as pd_series')
        self.assertEqual(base, 'from pandas.core import series as')
        self.assertEqual(specifier, ['pd_series'])


if __name__ == '__main__':
    unittest.main()
