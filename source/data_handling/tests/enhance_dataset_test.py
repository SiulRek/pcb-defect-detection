import os
import unittest
import tensorflow as tf
import numpy as np

from source.data_handling.manipulation.enhance_dataset import enhance_dataset
from source.utils import TestResultLogger


ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
OUTPUT_DIR = os.path.join(ROOT_DIR, "source", "data_handling", "tests", "outputs")
DATA_DIR = os.path.join(ROOT_DIR, "source", "data_handling", "tests", "data")
LOG_FILE = os.path.join(OUTPUT_DIR, "test_results.log")


class TestEnhanceDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cls.logger = TestResultLogger(LOG_FILE)
        cls.logger.log_title("Enhance Dataset Test")

    def setUp(self):
        # Create a sample dataset of 100 elements
        self.dataset = tf.data.Dataset.range(100)

    def tearDown(self):
        self.logger.log_test_outcome(self._outcome.result, self._testMethodName)

    def test_shuffling(self):
        # Check for changes in the dataset order
        original_first_element = next(iter(self.dataset)).numpy()
        enhanced = enhance_dataset(self.dataset, shuffle=True, random_seed=42)
        enhanced_first_element = next(iter(enhanced)).numpy()
        self.assertNotEqual(
            original_first_element,
            enhanced_first_element,
            "Shuffling did not change dataset order.",
        )

    def test_batching(self):
        enhanced = enhance_dataset(self.dataset, batch_size=10)
        self.assertEqual(
            enhanced.cardinality().numpy(),
            10,
            "Batching should create 10 batches from 100 elements.",
        )

    def test_repeating(self):
        repeated = enhance_dataset(self.dataset, repeat_num=3)
        self.assertEqual(
            repeated.cardinality().numpy(),
            300,
            "Dataset should be repeated 3 times, totaling 300 elements.",
        )

    def test_caching(self):
        try:
            enhanced = enhance_dataset(self.dataset, cache=True)
            self.assertIsInstance(
                enhanced, tf.data.Dataset, "Caching failed to return a tf.data.Dataset."
            )
        except Exception as e:
            self.fail(f"Enhance dataset with caching raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()