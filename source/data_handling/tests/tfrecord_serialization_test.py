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
RESTORED_DATA_DIR = os.path.join(OUTPUT_DIR, "restored_data")
LOG_FILE = os.path.join(OUTPUT_DIR, "test_results.log")
JPEG_TFRECORD_FILE = os.path.join(OUTPUT_DIR, "jpeg_test.tfrecord")
PNG_TFRECORD_FILE = os.path.join(OUTPUT_DIR, "png_test.tfrecord")


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

    def load_and_decode_image_with_float_labels(self, path, label):
        image_data = tf.io.read_file(path)
        image = tf.image.decode_image(image_data, channels=3)
        return image, tf.convert_to_tensor(label, dtype=tf.float32)

    def setUp(self):
        self.jpeg_data_dicts = [
            {"path": "figure_1.jpeg", "category_codes": 0},
            {"path": "figure_2.jpeg", "category_codes": 1},
            {"path": "figure_3.jpeg", "category_codes": 2},
        ]
        self.png_data_dicts = [
            {"path": "figure_4.png", "category_codes": 3},
            {"path": "figure_5.png", "category_codes": 4},
        ]
        for data_dict in self.jpeg_data_dicts + self.png_data_dicts:
            data_dict["path"] = os.path.join(DATA_DIR, data_dict["path"])

        self.jpeg_dataset = tf.data.Dataset.from_tensor_slices(
            (
                [data_dict["path"] for data_dict in self.jpeg_data_dicts],
                [data_dict["category_codes"] for data_dict in self.jpeg_data_dicts],
            )
        ).map(self.load_and_decode_image)

        self.png_dataset = tf.data.Dataset.from_tensor_slices(
            (
                [data_dict["path"] for data_dict in self.png_data_dicts],
                [data_dict["category_codes"] for data_dict in self.png_data_dicts],
            )
        ).map(self.load_and_decode_image)

    def tearDown(self):
        self.logger.log_test_outcome(self._outcome.result, self._testMethodName)
        if os.path.exists(JPEG_TFRECORD_FILE):
            os.remove(JPEG_TFRECORD_FILE)
        if os.path.exists(PNG_TFRECORD_FILE):
            os.remove(PNG_TFRECORD_FILE)

    def test_serialize_deserialize_jpeg(self):
        """Test the serialization and deserialization of a JPEG dataset."""
        serialize_dataset_to_tf_record(self.jpeg_dataset, JPEG_TFRECORD_FILE, "jpeg")
        restored_dataset = deserialize_dataset_from_tfrecord(
            JPEG_TFRECORD_FILE, tf.int8
        )

        for i, (original, restored) in enumerate(
            zip(self.jpeg_dataset, restored_dataset)
        ):
            original_image, original_label = original
            restored_image, restored_label = restored

            restored_image_path = os.path.join(
                RESTORED_DATA_DIR, f"restored_jpeg_{i}.jpeg"
            )
            tf.io.write_file(restored_image_path, tf.io.encode_jpeg(restored_image))

            self.assertTrue(
                np.allclose(original_image.numpy(), restored_image.numpy(), atol=100),
                "Restored images are not close enough to original images.",
            )
            self.assertEqual(
                original_label.numpy(),
                restored_label.numpy(),
                "Restored labels do not match original labels.",
            )

    def test_serialize_deserialize_png(self):
        """Test the serialization and deserialization of a PNG dataset."""
        serialize_dataset_to_tf_record(self.png_dataset, PNG_TFRECORD_FILE, "png")
        restored_dataset = deserialize_dataset_from_tfrecord(PNG_TFRECORD_FILE, tf.int8)

        for i, (original, restored) in enumerate(
            zip(self.png_dataset, restored_dataset)
        ):
            original_image, original_label = original
            restored_image, restored_label = restored

            restored_image_path = os.path.join(
                RESTORED_DATA_DIR, f"restored_png_{i}.png"
            )
            tf.io.write_file(restored_image_path, tf.io.encode_png(restored_image))

            self.assertTrue(
                np.allclose(original_image.numpy(), restored_image.numpy(), atol=1e-5),
                "Restored images are not close enough to original images.",
            )
            self.assertEqual(
                original_label.numpy(),
                restored_label.numpy(),
                "Restored labels do not match original labels.",
            )

    def test_serialize_deserialize_with_float_labels(self):
        """Test the serialization and deserialization of a dataset with floating-point labels."""
        float_label_data_dicts = [
            {"path": "figure_4.png", "category_codes": [0.1, 0.2]},
            {"path": "figure_5.png", "category_codes": [1.1, 1.2]},
        ]
        for data_dict in float_label_data_dicts:
            data_dict["path"] = os.path.join(DATA_DIR, data_dict["path"])

        float_label_dataset = tf.data.Dataset.from_tensor_slices(
            (
                [data_dict["path"] for data_dict in float_label_data_dicts],
                [data_dict["category_codes"] for data_dict in float_label_data_dicts],
            )
        ).map(self.load_and_decode_image_with_float_labels)

        float_tfrecord_file = os.path.join(OUTPUT_DIR, "float_label_test.tfrecord")
        serialize_dataset_to_tf_record(float_label_dataset, float_tfrecord_file, "png")
        restored_dataset = deserialize_dataset_from_tfrecord(
            float_tfrecord_file, tf.float32
        )

        for original, restored in zip(float_label_dataset, restored_dataset):
            original_image, original_label = original
            restored_image, restored_label = restored

            self.assertTrue(
                np.allclose(original_image.numpy(), restored_image.numpy(), atol=1e-5),
                "Restored images are not close enough to original images.",
            )
            self.assertTrue(
                np.allclose(original_label.numpy(), restored_label.numpy(), atol=1e-5),
                "Restored labels do not match original labels.",
            )

    def test_file_not_found_error_on_serialization(self):
        """Test if the correct exception is raised when the directory does not exist for
        serialization."""
        with self.assertRaises(FileNotFoundError):
            serialize_dataset_to_tf_record(
                self.jpeg_dataset,
                os.path.join(ROOT_DIR, "non_existent_directory", "test.tfrecord"),
                "jpeg",
            )

    def test_file_not_found_error_on_deserialization(self):
        """Test if the correct exception is raised when the TFRecord file does not exist for
        deserialization."""
        with self.assertRaises(FileNotFoundError):
            deserialize_dataset_from_tfrecord(
                os.path.join(ROOT_DIR, "non_existent_directory", "test.tfrecord"),
                tf.int8,
            )


if __name__ == "__main__":
    unittest.main()
