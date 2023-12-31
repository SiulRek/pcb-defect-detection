import os
import re
import unittest

from source.utils.randomly_select_sequential_keys import randomly_select_sequential_keys, is_sequential  
from source.utils.test_result_logger import TestResultLogger


ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..','..')
OUTPUT_DIR = os.path.join(ROOT_DIR, r'source/utils/tests/outputs')
LOG_FILE = os.path.join(OUTPUT_DIR, 'test_results.log')

#TODO: Make Test Suite more readable

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

    def get_stripped_dict_keys(self, input_dict, separator='__'):
        """ 
        Get the keys of a dictionary with the separator and part after the separator removed.
        
        Args:
        - input_dict (dict): The input dictionary.
        - separator (str, optional): The separator used in the key pattern. Defaults to '__'.
        
        Returns:
        - (list): A list of keys in the input dictionary.
        """
        return [key.split(separator)[0] for key in input_dict.keys()]

    def test_some_keys_not_matching(self):
        """ 
        Test that the function raises an error when only some keys do not match the pattern.
        """
        input_dict = {"name__I0": "value1", "B2": "value2"}
        with self.assertRaises(KeyError):
            randomly_select_sequential_keys(input_dict)

    def test_non_sequential_indices(self):
        """ 
        Test that the function raises an error when the indices are not sequential.
        """
        input_dict = {"name1__I1": "value1", "name2__I3": "value2"}
        with self.assertRaises(KeyError):
            randomly_select_sequential_keys(input_dict)

    def test_all_keys_matching(self):
        """ 
        Test that all keys are selected when all keys match the pattern and have different indices.
        """
        input_dict = {"name0__I0": "value0", "name1__I1": "value1", "name2__I2": "value2"}
        output_dict = randomly_select_sequential_keys(input_dict)
        stripped_input_keys = self.get_stripped_dict_keys(input_dict)
        self.assertTrue(all(key in stripped_input_keys for key in output_dict))
        self.assertEqual(len(output_dict), 3)
        self.assertTrue(is_sequential([int(key.split('name')[1]) for key in output_dict]))

    def test_normal_operation(self):
        """ 
        Test the normal operation of the function.
        """
        input_dict = {"a_name0__I0": "value0", "b_name0__I0": "alt0", "a_name1__I1": "value1", "b_name1__I1": "alt1"}
        output_dict = randomly_select_sequential_keys(input_dict)
        stripped_input_keys = self.get_stripped_dict_keys(input_dict)
        self.assertTrue(all(key in stripped_input_keys for key in output_dict))
        self.assertEqual(len(output_dict), 2)
        self.assertTrue(is_sequential([int(key.split('name')[1]) for key in output_dict]))
    
    def _generate_test_data(self, num_sequences):
        """
        Generate test data with sequential keys for testing.

        Args:
        - num_sequences (int): The number of sequential pairs to generate.

        Returns:
        - (dict): A dictionary with generated test data.
        """
        return {f"{i % 2}_name{i // 2}__I{i // 2}": f"value{i}" for i in range(num_sequences * 2)}

    def test_normal_operation_with_long_sequence(self):
        """ 
        Test the normal operation of the function with a longer sequence.
        """
        num_sequences = 111 
        input_dict = self._generate_test_data(num_sequences)
        output_dict = randomly_select_sequential_keys(input_dict)
        stripped_input_keys = self.get_stripped_dict_keys(input_dict)
        self.assertTrue(all(key in stripped_input_keys for key in output_dict))
        self.assertEqual(len(output_dict), num_sequences)
        self.assertTrue(is_sequential([int(key.split('name')[1]) for key in output_dict]))

    def test_resilient_operation_1(self):
        """ 
        Test that the function is resilient to unique identifier specified in keys.
        """
        input_dict = {"name1__1__I1": "value1", "name1__2__I1": "alt1", "name0__3__I0": "value0",  "name0__4__I0": "alt0"}
        output_dict = randomly_select_sequential_keys(input_dict)
        stripped_input_keys = ['name1__1', 'name1__2', 'name0__3', 'name0__4']
        self.assertTrue(all(key in stripped_input_keys for key in output_dict))
        self.assertEqual(len(output_dict), 2)
        self.assertTrue(is_sequential([int(key.split('name')[1][0]) for key in output_dict]))
    
    def test_resilient_operation_2(self):
        """ 
        Test that the function is resilient to the order of the keys.
        """
        input_dict = {"aname1__I1": "value1", "aname0__I0": "value0", "bname1__I1": "alt1", "bname0__I0": "alt0"}
        output_dict = randomly_select_sequential_keys(input_dict)
        stripped_input_keys = self.get_stripped_dict_keys(input_dict)
        self.assertTrue(all(key in stripped_input_keys for key in output_dict))
        self.assertEqual(len(output_dict), 2)
        self.assertTrue(is_sequential([int(key.split('name')[1]) for key in output_dict]))

    def test_keys_with_frequency_simple(self):
        """ 
        Test that keys with frequency specification are processed correctly.
        """
        input_dict = {
            "name1__I0": "value0", 
            "name2__I0F10": "alt0", 
            "name3__I1": "value1", 
            "name4__I1F10": "alt1",
            "name5__I2F10": "alt2",
        }
        output_dict = randomly_select_sequential_keys(input_dict)
        stripped_input_keys = self.get_stripped_dict_keys(input_dict)
        self.assertTrue(all(key in stripped_input_keys for key in output_dict))
        self.assertEqual(len(output_dict), 3)

    def test_keys_with_frequency_with_probability(self):
        """ 
        Test that keys with frequency specification are selected with the correct probability.
        """

        input_dict = {
            "name1__I0": "value0", 
            "name2__I0F10": "alt0", 
            "name3__I1": "value1", 
            "name4__I1F10": "alt1"
        }
        keys = ['name1', 'name2', 'name3', 'name4']
        output_dicts = []
        for _ in range(1000):
            output_dicts.append(randomly_select_sequential_keys(input_dict))
        
        key_counts = {}
        for key in keys:
            key_counts[key] = sum([1 if key in output_dict else 0 for output_dict in output_dicts])

        self.assertAlmostEqual(key_counts["name1"], 100, delta=40)
        self.assertAlmostEqual(key_counts["name3"], 100, delta=40)  
   
    def test_pattern_ending_allowed(self):
        """
        Test that keys with additional allowed characters after the pattern are correctly identified.
        """
        separator = '__'
        input_dict = {
            f"name1{separator}extra{separator}I0": "value0", 
            f"name2{separator}I1": "value1", 
            f"name3{separator}I2{separator}extra": "value2",  
            f"name4{separator}I3F10": "value3",  
            f"name5{separator}I4F10{separator}extra": "value4" 
        }

        expected_dict = {
            f"name1{separator}extra": "value0",
            f"name2": "value1",
            f"name3{separator}extra": "value2",
            f"name4": "value3",
            f"name5{separator}extra": "value4"
        }
      
        output_dict = randomly_select_sequential_keys(input_dict)
        self.assertEqual(output_dict, expected_dict)

    def test_pattern_ending_not_allowed(self):
        """
        Test that keys with additional not allowed characters after the pattern are correctly identified and lead to error.
        """
        separator = '__'
        input_dict = {
            f"name1{separator}I0": "value0", 
            f"name2{separator}I1": "value1", 
            f"name3{separator}I2_extra": "value2",  
            f"name4{separator}I3F10": "value3",  
            f"name5{separator}I4F10_extra": "value4" 
        }

        with self.assertRaises(KeyError):
            randomly_select_sequential_keys(input_dict)


if __name__ == '__main__':
    unittest.main()
