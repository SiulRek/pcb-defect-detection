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

    Note: 
    - All test cases take the default separator value of '__' for simplicity.
    - The naming convention for the dictionary key looks like this:
        {key identifier letter}_{key}_i{index}__I{index}F{frequency}__extra
        ° Only the 'key' is required, the rest are dependent on the test cases.
        ° __I{index}F{frequency} is the pattern that the function looks for.
        ° i{index} is used to verify the operation as the pattern __I{index}F{frequency}
        will be removed in the output dictionary.
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
        input_dict = {"a_key__I0": "value1", "b_key": "value2"}
        with self.assertRaises(KeyError):
            randomly_select_sequential_keys(input_dict)

    def test_non_sequential_indices(self):
        """ 
        Test that the function raises an error when the indices are not sequential.
        """
        input_dict = {"a_key_i1__I1": "value1", "b_key_i3__I3": "value2"}
        with self.assertRaises(KeyError):
            randomly_select_sequential_keys(input_dict)

    def test_all_keys_matching(self):
        """ 
        Test that all keys are selected when all keys match the pattern and have different indices.
        """
        input_dict = {"a_key_i0__I0": "value0", "b_key_i1__I1": "value1", "c_key_i2__I2": "value2"}
        output_dict = randomly_select_sequential_keys(input_dict)
        stripped_input_keys = self.get_stripped_dict_keys(input_dict)
        self.assertTrue(all(key in stripped_input_keys for key in output_dict))
        self.assertEqual(len(output_dict), 3)
        self.assertTrue(is_sequential([int(key.split('i')[1]) for key in output_dict]))

    def test_normal_operation(self):
        """ 
        Test the normal operation of the function.
        """
        input_dict = {"a_key_i0__I0": "value0", "b_key_i0__I0": "alt0", "a_key_i1__I1": "value1", "b_key_i1__I1": "alt1"}
        output_dict = randomly_select_sequential_keys(input_dict)
        stripped_input_keys = self.get_stripped_dict_keys(input_dict)
        self.assertTrue(all(key in stripped_input_keys for key in output_dict))
        self.assertEqual(len(output_dict), 2)
        self.assertTrue(is_sequential([int(key.split('i')[1]) for key in output_dict]))
    
    def _generate_test_data(self, num_sequences):
        """
        Generate test data with sequential keys for testing.

        Args:
        - num_sequences (int): The number of sequential pairs to generate.

        Returns:
        - (dict): A dictionary with generated test data.
        """
        return {f"{i % 2}_key_i{i // 2}__I{i // 2}": f"value{i}" for i in range(num_sequences * 2)}

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
        self.assertTrue(is_sequential([int(key.split('i')[1]) for key in output_dict]))

    def test_resilient_operation_1(self):
        """ 
        Test that the function is resilient to unique identifier specified in keys.
        """
        input_dict = {"key_i1__1__I1": "value1", "key_i1__2__I1": "alt1", "key_i0__3__I0": "value0",  "key_i0__4__I0": "alt0"}
        output_dict = randomly_select_sequential_keys(input_dict)
        stripped_input_keys = ['key_i1__1', 'key_i1__2', 'key_i0__3', 'key_i0__4']
        self.assertTrue(all(key in stripped_input_keys for key in output_dict))
        self.assertEqual(len(output_dict), 2)
        self.assertTrue(is_sequential([int(key.split('i')[1][0]) for key in output_dict]))
    
    def test_resilient_operation_2(self):
        """ 
        Test that the function is resilient to the order of the keys.
        """
        input_dict = {"a_key_i1__I1": "value1", "b_key_i0__I0": "value0", "c_key_i1__I1": "alt1", "d_key_i0__I0": "alt0"}
        output_dict = randomly_select_sequential_keys(input_dict)
        stripped_input_keys = self.get_stripped_dict_keys(input_dict)
        self.assertTrue(all(key in stripped_input_keys for key in output_dict))
        self.assertEqual(len(output_dict), 2)
        self.assertTrue(is_sequential([int(key.split('i')[1]) for key in output_dict]))
    
    def test_key_already_selected(self):
        """ 
        Test that the function is resilient to keys that have already been selected.
        """
        input_dict = {"a_key__I0": "value0", "a_key__I1": "value1"}
        
        with self.assertRaises(KeyError):
            randomly_select_sequential_keys(input_dict)

    def test_keys_with_frequency_simple(self):
        """ 
        Test that keys with frequency specification are processed correctly.
        """
        input_dict = {
            "a_key__I0": "value0", 
            "b_key__I0F10": "alt0", 
            "c_key__I1": "value1", 
            "d_key__I1F10": "alt1",
            "e_key__I2F10": "alt2",
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
            "a_key__I0": "value0", 
            "b_key__I0F10": "alt0", 
            "c_key__I1": "value1", 
            "d_key__I1F10": "alt1"
        }
        keys = ['a_key', 'b_key', 'c_key', 'd_key']
        output_dicts = []
        for _ in range(1000):
            output_dicts.append(randomly_select_sequential_keys(input_dict))
        
        key_counts = {}
        for key in keys:
            key_counts[key] = sum([1 if key in output_dict else 0 for output_dict in output_dicts])

        self.assertAlmostEqual(key_counts["a_key"], 91, delta=25)
        self.assertAlmostEqual(key_counts["c_key"], 91, delta=25)  
   
    def test_pattern_ending_allowed(self):
        """
        Test that keys with additional allowed characters after the pattern are correctly identified.
        """
        separator = '__'
        input_dict = {
            f"a_key{separator}extra{separator}I0": "value0", 
            f"b_key{separator}I1": "value1", 
            f"c_key{separator}I2{separator}extra": "value2",  
            f"d_key{separator}I3F10": "value3",  
            f"e_key{separator}I4F10{separator}extra": "value4" 
        }

        expected_dict = {
            f"a_key{separator}extra": "value0",
            f"b_key": "value1",
            f"c_key{separator}extra": "value2",
            f"d_key": "value3",
            f"e_key{separator}extra": "value4"
        }
      
        output_dict = randomly_select_sequential_keys(input_dict)
        self.assertEqual(output_dict, expected_dict)

    def test_pattern_ending_not_allowed(self):
        """
        Test that keys with additional not allowed characters after the pattern are correctly identified and lead to error.
        """
        separator = '__'
        input_dict = {
            f"a_key{separator}I0": "value0", 
            f"b_key{separator}I1": "value1", 
            f"c_key{separator}I2_extra": "value2",  
            f"d_key{separator}I3F10": "value3",  
            f"e_key{separator}I4F10_extra": "value4" 
        }

        with self.assertRaises(KeyError):
            randomly_select_sequential_keys(input_dict)


if __name__ == '__main__':
    unittest.main()
