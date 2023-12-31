import os
import re
import unittest

from source.utils.randomly_select_sequential_keys import randomly_select_sequential_keys, is_sequential  
from source.utils.test_result_logger import TestResultLogger


ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..','..')
OUTPUT_DIR = os.path.join(ROOT_DIR, r'source/utils/tests/outputs')
LOG_FILE = os.path.join(OUTPUT_DIR, 'test_results.log')


class TestRandomlySelectSequentialKeys(unittest.TestCase):
    """
    Unit tests for `randomly_select_sequential_keys`.

    This suite tests the accuracy of the function in identifying and handling sequential key 
    patterns in dictionaries. It covers various cases, including invalid patterns, sequential 
    integrity, and frequency-based key selection. Each test ensures the function's robustness
    and error handling, verifying its consistency across different key configurations and
    input scenarios.

    Note: All test cases take the default separator value of '__' for simplicity.
    """


    @classmethod
    def setUpClass(cls):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cls.logger = TestResultLogger(LOG_FILE, 'Randomly Select Sequential Keys')

    def tearDown(self):
        self.logger.log_test_outcome(self._outcome.result, self._testMethodName)
    
    def test_no_matching_keys(self):
        """ 
        Test that the function returns the input dictionary when no keys match the pattern.
        """
        input_dict = {"A1": "value1", "B2": "value2"}
        self.assertEqual(randomly_select_sequential_keys(input_dict), input_dict)

    def test_some_keys_not_matching(self):
        """ 
        Test that the function raises an error when only some keys do not match the pattern.
        """
        input_dict = {"name__L0": "value1", "B2": "value2"}
        with self.assertRaises(KeyError):
            randomly_select_sequential_keys(input_dict)

    def test_non_sequential_indices(self):
        """ 
        Test that the function raises an error when the indices are not sequential.
        """
        input_dict = {"name__L1": "value1", "name__L3": "value2"}
        with self.assertRaises(KeyError):
            randomly_select_sequential_keys(input_dict)

    def test_all_keys_matching(self):
        """ 
        Test that all keys are selected when all keys match the pattern and have different indices.
        """
        input_dict = {"name__L0": "value0", "name__L1": "value1", "name__L2": "value2"}
        output_dict = randomly_select_sequential_keys(input_dict)
        self.assertTrue(all(key in input_dict for key in output_dict))
        self.assertEqual(len(output_dict), 3)
        self.assertTrue(is_sequential([int(key.split('name__L')[1]) for key in output_dict]))

    def test_normal_operation(self):
        """ 
        Test the normal operation of the function.
        """
        input_dict = {"name1__L0": "value0", "name2__L0": "alt0", "name1__L1": "value1", "name2__L1": "alt1"}
        output_dict = randomly_select_sequential_keys(input_dict)
        self.assertTrue(all(key in input_dict for key in output_dict))
        self.assertEqual(len(output_dict), 2)
        extracted_indices = []
        for key in output_dict:
            match = re.search(r'L(\d+)', key)
            if match:
                extracted_indices.append(int(match.group(1)))
        self.assertTrue(is_sequential(extracted_indices))

    def test_resilient_operation_1(self):
        """ 
        Test that the function is resilient to unique identifier specified in keys.
        """
        input_dict = {"name1__1__L1": "value1", "name2__1__L1": "alt1", "name1__2__L0": "value0",  "name2__2__L0": "alt0"}
        output_dict = randomly_select_sequential_keys(input_dict)
        self.assertTrue(all(key in input_dict for key in output_dict))
        self.assertEqual(len(output_dict), 2)
        extracted_indices = []
        for key in output_dict:
            match = re.search(r'L(\d+)', key)
            if match:
                extracted_indices.append(int(match.group(1)))
        self.assertTrue(is_sequential(extracted_indices))
    
    def test_resilient_operation_2(self):
        """ 
        Test that the function is resilient to the order of the keys.
        """
        input_dict = {"name1__L1": "value1", "name1__L0": "value0", "name2__L1": "alt1", "name2__L0": "alt0"}
        output_dict = randomly_select_sequential_keys(input_dict)
        self.assertTrue(all(key in input_dict for key in output_dict))
        self.assertEqual(len(output_dict), 2)
        extracted_indices = []
        for key in output_dict:
            match = re.search(r'L(\d+)', key)
            if match:
                extracted_indices.append(int(match.group(1)))
        self.assertTrue(is_sequential(extracted_indices))

    def test_keys_with_frequency_simple(self):
        """ 
        Test that keys with frequency specification are processed correctly.
        """
        input_dict = {
            "name__L0": "value0", 
            "name__L0F10": "alt0", 
            "name__L1": "value1", 
            "name__L1F10": "alt1",
            "name__L2F10": "alt2",
        }
        output_dict = randomly_select_sequential_keys(input_dict)
        self.assertTrue(all(key in input_dict for key in output_dict))
        self.assertEqual(len(output_dict), 3)

    def test_keys_with_frequency_with_probability(self):
        """ 
        Test that keys with frequency specification are selected with the correct probability.
        """

        input_dict = {
            "name__L0": "value0", 
            "name__L0F10": "alt0", 
            "name__L1": "value1", 
            "name__L1F10": "alt1"
        }
        output_dicts = []
        for _ in range(1000):
            output_dicts.append(randomly_select_sequential_keys(input_dict))
        
        key_counts = {}
        for key in input_dict:
            key_counts[key] = sum([1 if key in output_dict else 0 for output_dict in output_dicts])

        self.assertAlmostEqual(key_counts["name__L0"], 100, delta=40)
        self.assertAlmostEqual(key_counts["name__L1"], 100, delta=40)  

    def test_pattern_ending(self):
        """
        Test that keys with additional characters after the pattern are correctly identified and lead to error.
        """
        separator = '__'
        input_dict = {
            f"name{separator}L0": "value0", 
            f"name{separator}L1": "value1", 
            f"name{separator}L2_extra": "value2",  
            f"name{separator}L3F10": "value3",  
            f"name{separator}L4F10_extra": "value4" 
        }

        with self.assertRaises(KeyError):
            randomly_select_sequential_keys(input_dict)


if __name__ == '__main__':
    unittest.main()
