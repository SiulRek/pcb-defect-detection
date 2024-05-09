import unittest

from temporary_folder.tasks.helpers.for_cleanup.split_import_statement import (
    split_import_statement,
)


class TestSplitImportStatement(unittest.TestCase):
    def test_simple_import(self):
        base, original_names, alias_names = split_import_statement("import os")
        self.assertEqual(base, "import")
        self.assertEqual(original_names, ["os"])
        self.assertEqual(alias_names, ["os"])

    def test_import_with_alias(self):
        base, original_names, alias_names = split_import_statement("import numpy as np")
        self.assertEqual(base, "import")
        self.assertEqual(original_names, ["numpy"])
        self.assertEqual(alias_names, ["np"])

    def test_from_import_single(self):
        base, original_names, alias_names = split_import_statement("from sys import path")
        self.assertEqual(base, "from sys import")
        self.assertEqual(original_names, ["path"])
        self.assertEqual(alias_names, ["path"])

    def test_from_import_multiple(self):
        base, original_names, alias_names = split_import_statement(
            "from datetime import datetime, timedelta"
        )
        self.assertEqual(base, "from datetime import")
        self.assertEqual(original_names, ["datetime", "timedelta"])
        self.assertEqual(alias_names, ["datetime", "timedelta"])

    def test_from_import_with_parentheses(self):
        base, original_names, alias_names = split_import_statement(
            "from math import (sqrt, ceil, floor)"
        )
        self.assertEqual(base, "from math import")
        self.assertEqual(original_names, ["sqrt", "ceil", "floor"])
        self.assertEqual(alias_names, ["sqrt", "ceil", "floor"])

    def test_from_import_with_alias(self):
        base, original_names, alias_names = split_import_statement(
            "from pandas.core import series as pd_series"
        )
        self.assertEqual(base, "from pandas.core import")
        self.assertEqual(original_names, ["series"])
        self.assertEqual(alias_names, ["pd_series"])

    def test_complex_import(self):
        base, original_names, alias_names = split_import_statement(
            "from math import (\n sqrt as s,\n ceil as c,\n floor as f\n)"
        )
        self.assertEqual(base, "from math import")
        self.assertEqual(original_names, ["sqrt", "ceil", "floor"])
        self.assertEqual(alias_names, ["s", "c", "f"])


if __name__ == "__main__":
    unittest.main()
