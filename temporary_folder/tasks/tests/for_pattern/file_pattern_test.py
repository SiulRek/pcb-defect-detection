import unittest
import re

FILE_TAG = "File"
FILE_PATTERN = re.compile(rf"#\s*((?:\S+\.(?:py|txt|log|md|csv))\s*(?:,\s*\S+\.(?:py|txt|log|md|csv)\s*)*|{FILE_TAG})")


class TestFilePatternRegex(unittest.TestCase):
    def test_single_file(self):
        test_string = "# file_1.py"
        match = FILE_PATTERN.search(test_string)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), 'file_1.py')

    def test_multiple_files(self):
        test_string = "# file_1.py, dir\\file_2.txt, file_3.log"
        match = FILE_PATTERN.search(test_string)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), 'file_1.py, dir\\file_2.txt, file_3.log')

    def test_files_with_spaces(self):
        test_string = "# file_1.py ,  file_2.txt , dir/dir/file_3.log"
        match = FILE_PATTERN.search(test_string)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), 'file_1.py ,  file_2.txt , dir/dir/file_3.log')
    
    def test_file_with_file_tag(self):
        test_string = "# File"
        match = FILE_PATTERN.search(test_string)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), 'File')

    def test_no_files(self):
        test_string = "# Just a comment without a file"
        match = FILE_PATTERN.search(test_string)
        self.assertIsNone(match)

    def test_files_with_varied_extensions(self):
        test_string = "# project.csv, report.md, log_file.log"
        match = FILE_PATTERN.search(test_string)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), 'project.csv, report.md, log_file.log')

    def test_invalid_extensions(self):
        test_string = "# image.aux, video.mp4"
        match = FILE_PATTERN.search(test_string)
        self.assertIsNone(match)

    def test_complex_spacing_and_formatting(self):
        test_string = "#   file_1.py,file_2.py,    file_3.py "
        match = FILE_PATTERN.search(test_string)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), 'file_1.py,file_2.py,    file_3.py ')

if __name__ == '__main__':
    unittest.main()
