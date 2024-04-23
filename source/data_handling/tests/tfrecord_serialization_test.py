import os
import unittest
import tensorflow as tf
import numpy as np

from source.data_handling.tfrecord_serialization.serialize import (
    serialize_dataset_to_tf_record,
)
from source.data_handling.tfrecord_serialization.deserialize import (
    deserialize_dataset_from_tfrecord,
)
from source.utils import TestResultLogger

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
OUTPUT_DIR = os.path.join(ROOT_DIR, "source", "data_handling", "tests", "outputs")
DATA_DIR = os.path.join(ROOT_DIR, "source", "data_handling", "tests", "data")
LOG_FILE = os.path.join(OUTPUT_DIR, "test_results.log")
TFRECORD_FILE = os.path.join(OUTPUT_DIR, "test.tfrecord")


class TestTFRecordHandling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cls.logger = TestResultLogger(LOG_FILE)
        cls.logger.log_title("TFRecord Serialization and Deserialization Test")

    def load_and_decode_image(self, path, label):
        image_data = tf.io.read_file(path)
        image = tf.image.decode_image(image_data, channels=3)
        return image, tf.cast(label, tf.int8)

    def setUp(self):
        # Load images as byte strings and then decode
        self.data_dicts = [
            {"path": "figure_1.jpeg", "category_codes": 0},
            {"path": "figure_2.jpeg", "category_codes": 1},
            {"path": "figure_3.jpeg", "category_codes": 2},
            # {"path": "figure_4.png", "category_codes": 3},
            # {"path": "figure_5.png", "category_codes": 4},
        ]
        for data_dict in self.data_dicts:
            data_dict["path"] = os.path.join(DATA_DIR, data_dict["path"])

        # Create a dataset that loads the images
        self.dataset = tf.data.Dataset.from_tensor_slices(
            (
                [data_dict["path"] for data_dict in self.data_dicts],
                [data_dict["category_codes"] for data_dict in self.data_dicts],
            )
        ).map(self.load_and_decode_image)

    def tearDown(self):
        self.logger.log_test_outcome(self._outcome.result, self._testMethodName)
        if os.path.exists(TFRECORD_FILE):
            os.remove(TFRECORD_FILE)

    def test_serialize_deserialize(self):
        """Test the serialization and deserialization of a dataset."""
        serialize_dataset_to_tf_record(self.dataset, TFRECORD_FILE)
        restored_dataset = deserialize_dataset_from_tfrecord(TFRECORD_FILE)

        for original, restored in zip(self.dataset, restored_dataset):
            original_image, original_label = original
            restored_image, restored_label = restored

            # Check if the images are close enough after serialization and JPEG compression
            self.assertTrue(
                np.allclose(original_image.numpy(), restored_image.numpy(), atol=1e-5),
                "Restored images are not close enough to original images.",
            )
            # Check if labels match exactly
            self.assertEqual(
                original_label.numpy(),
                restored_label.numpy(),
                "Restored labels do not match original labels.",
            )

    def test_file_not_found_error_on_serialization(self):
        """Test if the correct exception is raised when the directory does not exist for serialization."""
        with self.assertRaises(FileNotFoundError):
            serialize_dataset_to_tf_record(
                self.dataset,
                os.path.join(ROOT_DIR, "non_existent_directory", "test.tfrecord"),
            )

    def test_file_not_found_error_on_deserialization(self):
        """Test if the correct exception is raised when the TFRecord file does not exist for deserialization."""
        with self.assertRaises(FileNotFoundError):
            deserialize_dataset_from_tfrecord(
                os.path.join(ROOT_DIR, "non_existent_directory", "test.tfrecord")
            )


if __name__ == "__main__":
    unittest.main()
