import os
import re
import unittest

from source.utils.randomly_select_sequential_keys import randomly_select_sequential_keys, is_sequential  
from source.utils.test_result_logger import TestResultLogger


ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..','..')
OUTPUT_DIR = os.path.join(ROOT_DIR, r'source/utils/tests/outputs')
LOG_FILE = os.path.join(OUTPUT_DIR, 'test_results.log')

class TestRandomlySelectSequentialKeys(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cls.logger = TestResultLogger(LOG_FILE, 'Recursive Type Conversion Test')

    def tearDown(self):
        self.logger.log_test_outcome(self._outcome.result, self._testMethodName)
    
    def test_no_matching_keys(self):
        input_dict = {"A1": "value1", "B2": "value2"}
        self.assertEqual(randomly_select_sequential_keys(input_dict), input_dict)

    def test_some_keys_not_matching(self):
        input_dict = {"name__L1": "value1", "B2": "value2"}
        with self.assertRaises(ValueError):
            randomly_select_sequential_keys(input_dict)

    def test_non_sequential_indices(self):
        input_dict = {"name__L1": "value1", "name__L3": "value2"}
        with self.assertRaises(ValueError):
            randomly_select_sequential_keys(input_dict)

    def test_all_keys_matching(self):
        input_dict = {"name__L0": "value0", "name__L1": "value1", "name__L2": "value2"}
        output_dict = randomly_select_sequential_keys(input_dict)
        self.assertTrue(all(key in input_dict for key in output_dict))
        self.assertEqual(len(output_dict), 3)
        self.assertTrue(is_sequential([int(key.split('name__L')[1]) for key in output_dict]))

    def test_normal_operation(self):
        input_dict = {"name__L0": "value0", "name__L0_alt": "alt0", "name__L1": "value1", "name__L1_alt": "alt1"}
        output_dict = randomly_select_sequential_keys(input_dict)
        self.assertTrue(all(key in input_dict for key in output_dict))
        self.assertEqual(len(output_dict), 2)
        extracted_indices = []
        for key in output_dict:
            match = re.search(r'L(\d+)', key)
            if match:
                extracted_indices.append(int(match.group(1)))
        self.assertTrue(is_sequential(extracted_indices))


if __name__ == '__main__':
    unittest.main()
