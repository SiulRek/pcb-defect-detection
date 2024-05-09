import re
import unittest

RUN_SCRIPT_PATTERN = re.compile(r"#run (\S+\.py)")


class TestScriptNamePattern(unittest.TestCase):
    def test_valid_script_name(self):
        test_string = "#run python_script.py"
        match = RUN_SCRIPT_PATTERN.search(test_string)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), 'python_script.py')

    def test_invalid_script_name_no_py(self):
        test_string = "#run python_script.txt"
        match = RUN_SCRIPT_PATTERN.search(test_string)
        self.assertIsNone(match)

    def test_valid_script_name_with_path(self):
        test_string = "#run folder/subfolder/python_script.py"
        match = RUN_SCRIPT_PATTERN.search(test_string)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), 'folder/subfolder/python_script.py')


if __name__ == '__main__':
    unittest.main()
