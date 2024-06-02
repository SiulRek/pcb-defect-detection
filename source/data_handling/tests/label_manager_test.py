import os
import unittest

from source.data_handling.helpers.label_manager import LabelManager
from source.utils import TestResultLogger
import tensorflow as tf

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
OUTPUT_DIR = os.path.join(ROOT_DIR, "source", "data_handling", "tests", "outputs")
LOG_FILE = os.path.join(OUTPUT_DIR, "test_results.log")


class TestLabelManager(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cls.logger = TestResultLogger(LOG_FILE)
        cls.logger.log_title("Label Manager Test")

    def setUp(self):
        self.category_code_sample = {"category_codes": 2}
        self.invalid_labels = {"wrong_key": 3}

    def tearDown(self):
        self.logger.log_test_outcome(self._outcome.result, self._testMethodName)

    def test_categorical_labels_valid_input(self):
        manager = LabelManager("category_codes", num_classes=4)
        result = manager.get_label(self.category_code_sample)
        expected = tf.constant([0, 0, 1, 0], dtype=tf.float32)
        self.assertTrue(
            tf.reduce_all(tf.equal(result, expected)),
            "The categorical labels do not match expected output.",
        )

    def test_categorical_labels_invalid_key(self):
        manager = LabelManager("category_codes", num_classes=4)
        with self.assertRaises(KeyError):
            manager.get_label(self.invalid_labels)

    def test_sparse_categorical_labels_valid_input(self):
        manager = LabelManager("sparse_category_codes")
        result = manager.get_label(self.category_code_sample)
        expected = tf.constant(2, dtype=tf.int8)
        self.assertTrue(
            tf.reduce_all(tf.equal(result, expected)),
            "The sparse categorical labels do not match expected output.",
        )

    def test_sparse_categorical_labels_invalid_key(self):
        manager = LabelManager("sparse_category_codes")
        with self.assertRaises(KeyError):
            manager.get_label(self.invalid_labels)

    def test_object_detection_labels_not_implemented(self):
        manager = LabelManager("object_detection")
        with self.assertRaises(NotImplementedError):
            manager.get_label(self.category_code_sample)


if __name__ == "__main__":
    unittest.main()
