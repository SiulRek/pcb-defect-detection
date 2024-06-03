import os
import unittest

import numpy as np
import tensorflow as tf

from source.data_handling.io.load_dataset import load_dataset
from source.utils import TestResultLogger

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
OUTPUT_DIR = os.path.join(ROOT_DIR, "source", "data_handling", "tests", "outputs")
DATA_DIR = os.path.join(ROOT_DIR, "source", "data_handling", "tests", "data")
LOG_FILE = os.path.join(OUTPUT_DIR, "test_results.log")


class TestLoadDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cls.logger = TestResultLogger(LOG_FILE)
        cls.logger.log_title("Dataset Test")

        # Check if pandas is installed and import it if available
        try:
            import pandas as pd

            cls.pandas_installed = True
            cls.pd = pd
        except ImportError:
            cls.pandas_installed = False

    def setUp(self):
        self.data_dicts = [
            {"path": "figure_1.jpeg", "label": 0},
            {"path": "figure_2.jpeg", "label": 1},
            {"path": "figure_3.jpeg", "label": 2},
            {"path": "figure_4.png", "label": 3},
            {"path": "figure_5.png", "label": 4},
        ]
        for data_dict in self.data_dicts:
            data_dict["path"] = os.path.join(DATA_DIR, data_dict["path"])

    def tearDown(self):
        self.logger.log_test_outcome(self._outcome.result, self._testMethodName)

    def test_dataset_from_dicts(self):
        """ Test dataset creation from a list of dictionaries. """
        dataset = load_dataset(self.data_dicts, "sparse_category_codes")
        expected_labels = [0, 1, 2, 3, 4]
        for (img, label), expected_label in zip(dataset, expected_labels):
            self.assertIsInstance(img, tf.Tensor, "Image should be a Tensor.")
            self.assertEqual(
                label.numpy(), expected_label, "Label does not match expected."
            )

    def test_dataset_from_dataframe(self):
        """ Test dataset creation from a pandas DataFrame if pandas is installed. """
        if self.pandas_installed:
            df = self.pd.DataFrame(self.data_dicts)
            dataset = load_dataset(df, "sparse_category_codes")
            expected_labels = [0, 1, 2, 3, 4]
            for (img, label), expected_label in zip(dataset, expected_labels):
                self.assertIsInstance(img, tf.Tensor, "Image should be a Tensor.")
                self.assertEqual(
                    label.numpy(), expected_label, "Label does not match expected."
                )
        else:
            self.skipTest("pandas is not installed.")

    def test_one_hot_encoding(self):
        """ Test one-hot encoding for category codes. """
        num_classes = 5  # Assuming there are 5 classes for the one-hot encoding
        dataset = load_dataset(
            self.data_dicts, "category_codes", num_classes=num_classes
        )
        expected_labels = [0, 1, 2, 3, 4]
        for (_, label), expected_label in zip(dataset, expected_labels):
            expected_one_hot = tf.one_hot(expected_label, depth=num_classes)
            self.assertTrue(
                np.array_equal(label.numpy(), expected_one_hot.numpy()),
                "One-hot encoded labels do not match expected.",
            )

    def test_invalid_data_type(self):
        """ Test if ValueError is raised for invalid data type. """
        with self.assertRaises(ValueError):
            load_dataset("invalid_data_type", "category_codes")


if __name__ == "__main__":
    unittest.main()
