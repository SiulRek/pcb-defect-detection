import os
import unittest

import tensorflow as tf

from source.data_handling.helpers.label_manager import LabelManager
from source.utils import TestResultLogger

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
        self.binary_label = 1
        self.invalid_binary_label = 2
        self.categorical_label = 2
        self.invalid_label_key = {"wrong_key": 3}

    def tearDown(self):
        self.logger.log_test_outcome(self._outcome.result, self._testMethodName)

    def test_binary_labels_valid_input(self):
        manager = LabelManager("binary")
        result = manager.encode_label(self.binary_label)
        expected = tf.constant(1, dtype=tf.float32)
        self.assertTrue(
            tf.reduce_all(tf.equal(result, expected)),
            "The binary labels do not match expected output.",
        )

    def test_binary_labels_invalid_value(self):
        manager = LabelManager("binary")
        with self.assertRaises(ValueError):
            manager.encode_label(self.invalid_binary_label)

    def test_categorical_labels_valid_input(self):
        manager = LabelManager("category_codes", category_names=["a", "b", "c", "d"])
        result = manager.encode_label(self.categorical_label)
        expected = tf.constant([0, 0, 1, 0], dtype=tf.float32)
        self.assertTrue(
            tf.reduce_all(tf.equal(result, expected)),
            "The categorical labels do not match expected output.",
        )

    def test_sparse_categorical_labels_valid_input(self):
        manager = LabelManager(
            "sparse_category_codes", category_names=["a", "b", "c", "d"]
        )
        result = manager.encode_label(self.categorical_label)
        expected = tf.constant(2, dtype=tf.float32)
        self.assertTrue(
            tf.reduce_all(tf.equal(result, expected)),
            "The sparse categorical labels do not match expected output.",
        )

    def test_object_detection_labels_not_implemented(self):
        manager = LabelManager("object_detection")
        with self.assertRaises(NotImplementedError):
            manager.encode_label(self.categorical_label)

    def test_label_dtype(self):
        manager = LabelManager("category_codes", category_names=["a", "b", "c", "d"])
        self.assertEqual(
            manager.label_dtype,
            tf.float32,
            "Label dtype for category_codes should be tf.float32",
        )
        manager = LabelManager(
            "sparse_category_codes", category_names=["a", "b", "c", "d"]
        )
        self.assertEqual(
            manager.label_dtype,
            tf.float32,
            "Label dtype for sparse_category_codes should be tf.float32",
        )

        manager = LabelManager("object_detection")
        self.assertEqual(
            manager.label_dtype,
            tf.float32,
            "Label dtype for object_detection should be tf.float32",
        )

        manager = LabelManager("binary")
        self.assertEqual(
            manager.label_dtype,
            tf.float32,
            "Label dtype for binary should be tf.float32",
        )

    def test_label_conversion_dtype(self):
        manager = LabelManager("category_codes", category_names=["a", "b", "c", "d"])
        result = manager.encode_label(self.categorical_label)
        self.assertEqual(
            result.dtype,
            tf.float32,
            "The dtype of the encoded categorical label should be tf.float32",
        )

        manager = LabelManager(
            "category_codes", category_names=["a", "b", "c", "d"], dtype=tf.int32
        )
        result = manager.encode_label(self.categorical_label)
        self.assertEqual(
            result.dtype,
            tf.int32,
            "The dtype of the encoded categorical label should be tf.int32",
        )

        manager = LabelManager(
            "sparse_category_codes", category_names=["a", "b", "c", "d"]
        )
        result = manager.encode_label(self.categorical_label)
        self.assertEqual(
            result.dtype,
            tf.float32,
            "The dtype of the encoded sparse categorical label should be tf.float32",
        )

        manager = LabelManager(
            "sparse_category_codes", category_names=["a", "b", "c", "d"], dtype=tf.int32
        )
        result = manager.encode_label(self.categorical_label)
        self.assertEqual(
            result.dtype,
            tf.int32,
            "The dtype of the encoded sparse categorical label should be tf.int32",
        )

        manager = LabelManager("binary")
        result = manager.encode_label(self.binary_label)
        self.assertEqual(
            result.dtype,
            tf.float32,
            "The dtype of the encoded binary label should be tf.float32",
        )

        manager = LabelManager("binary", dtype=tf.int32)
        result = manager.encode_label(self.binary_label)
        self.assertEqual(
            result.dtype,
            tf.int32,
            "The dtype of the encoded binary label should be tf.int32",
        )

        manager = LabelManager("object_detection")
        with self.assertRaises(NotImplementedError):
            manager.encode_label(self.categorical_label)

    def test_convert_to_numeric(self):
        manager = LabelManager("category_codes", category_names=["a", "b", "c", "d"])
        self.assertEqual(
            manager.convert_to_numeric("c"),
            2,
            "Conversion to numeric failed for valid string label.",
        )
        self.assertEqual(
            manager.convert_to_numeric(3),
            3,
            "Conversion to numeric failed for numeric input.",
        )
        self.assertEqual(
            manager.convert_to_numeric(tf.constant(3)),
            3,
            "Conversion to numeric failed for tensor input.",
        )
        with self.assertRaises(ValueError):
            manager.convert_to_numeric("e")

    def test_decode_label(self):
        manager = LabelManager("category_codes", category_names=["a", "b", "c", "d"])
        self.assertEqual(
            manager.decode_label(2),
            "c",
            "Decoding failed for valid numeric label.",
        )
        self.assertEqual(
            manager.decode_label(tf.constant(2)),
            "c",
            "Decoding failed for valid tensor label.",
        )
        with self.assertRaises(ValueError):
            manager.decode_label(5)
        with self.assertRaises(ValueError):
            manager.decode_label("c")


if __name__ == "__main__":
    unittest.main()
