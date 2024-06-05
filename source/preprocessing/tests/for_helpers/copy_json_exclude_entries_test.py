import json
import os
import unittest

from source.preprocessing.helpers.for_tests.copy_json_exclude_entries import (
    copy_json_exclude_entries,
)
from source.testing.base_test_case import BaseTestCase


class TestCopyJsonExcludeEntries(BaseTestCase):
    """
    Test cases for the copy_json_exclude_entries function.

    The Test Suite contains the following test cases:
        - test_copy_json_exclude_entries: Tests the
            copy_json_exclude_entries function with a non-empty exclude_keys
            list.
        - test_copy_json_exclude_entries_empty_exclude_keys: Tests the
            copy_json_exclude_entries function with an empty exclude_keys list.
        - test_copy_json_exclude_entries_not_existing_key: Tests the
            copy_json_exclude_entries function with a non-existing key in the
            exclude_keys list.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()  

        cls.source_file = os.path.join(cls.output_dir, "source.json")
        cls.dest_file = os.path.join(cls.output_dir, "dest.json")

    def setUp(self):
        super().setUp()  

        source_data = {"key1": "value1", "key2": "value2", "key3": "value3"}
        with open(self.source_file, "w") as file:
            json.dump(source_data, file)

    def tearDown(self):
        super().tearDown()  

        if os.path.exists(self.source_file):
            os.remove(self.source_file)
        if os.path.exists(self.dest_file):
            os.remove(self.dest_file)

    def test_copy_json_exclude_entries(self):
        exclude_keys = ["key2"]
        copy_json_exclude_entries(self.source_file, self.dest_file, exclude_keys)

        with open(self.dest_file, "r") as file:
            dest_data = json.load(file)

        self.assertNotIn("key2", dest_data)
        self.assertIn("key1", dest_data)
        self.assertIn("key3", dest_data)

    def test_copy_json_exclude_entries_empty_exclude_keys(self):
        exclude_keys = []
        copy_json_exclude_entries(self.source_file, self.dest_file, exclude_keys)

        with open(self.dest_file, "r") as file:
            dest_data = json.load(file)

        with open(self.source_file, "r") as file:
            source_data = json.load(file)
        self.assertEqual(dest_data, source_data)

    def test_copy_json_exclude_entries_not_existing_key(self):
        exclude_keys = ["key4"]
        copy_json_exclude_entries(self.source_file, self.dest_file, exclude_keys)

        with open(self.dest_file, "r") as file:
            dest_data = json.load(file)

        with open(self.source_file, "r") as file:
            source_data = json.load(file)
        self.assertEqual(dest_data, source_data)


if __name__ == "__main__":
    unittest.main()