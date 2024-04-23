import os
import unittest
import tensorflow as tf
import numpy as np

from source.data_handling.manipulation.split_dataset import split_dataset
from source.utils import TestResultLogger

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
OUTPUT_DIR = os.path.join(ROOT_DIR, "source", "data_handling", "tests", "outputs")
DATA_DIR = os.path.join(ROOT_DIR, "source", "data_handling", "tests", "data")
LOG_FILE = os.path.join(OUTPUT_DIR, "test_results.log")

class TestDatasetFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cls.logger = TestResultLogger(LOG_FILE)
        cls.logger.log_title("TensorFlow Dataset Enhancement and Splitting Test")

    def setUp(self):
        self.dataset = tf.data.Dataset.range(100)

    def tearDown(self):
        self.logger.log_test_outcome(self._outcome.result, self._testMethodName)

    def test_split_proportions(self):
        train, val, test = split_dataset(self.dataset)
        self.assertEqual(train.cardinality().numpy(), 80, "Training set size should be 80% of total.")
        self.assertEqual(val.cardinality().numpy(), 10, "Validation set size should be 10% of total.")
        self.assertEqual(test.cardinality().numpy(), 10, "Test set size should be 10% of total.")

    def test_split_proportions_error(self):
        with self.assertRaises(ValueError):
            split_dataset(self.dataset, train_size=0.7, val_size=0.2, test_size=0.2)


if __name__ == "__main__":
    unittest.main()
