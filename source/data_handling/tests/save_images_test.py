import os
import unittest

import tensorflow as tf

from source.data_handling.io.save_images import save_images
from source.utils import TestResultLogger

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
OUTPUT_DIR = os.path.join(ROOT_DIR, "source", "data_handling", "tests", "outputs")
DATA_DIR = os.path.join(ROOT_DIR, "source", "data_handling", "tests", "data")
LOG_FILE = os.path.join(OUTPUT_DIR, "test_results.log")


class TestSaveImages(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cls.logger = TestResultLogger(LOG_FILE)
        cls.logger.log_title("Save Images Test")

    def load_and_decode_image(self, path, label):
        image_data = tf.io.read_file(path)
        image = tf.image.decode_image(image_data, channels=3)
        return image, tf.cast(label, tf.int8)

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

        self.test_output_dir = os.path.join(OUTPUT_DIR, "test_save_images")
        os.makedirs(self.test_output_dir, exist_ok=True)

    def tearDown(self):
        for file_name in os.listdir(self.test_output_dir):
            file_path = os.path.join(self.test_output_dir, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(self.test_output_dir)
        self.logger.log_test_outcome(self._outcome.result, self._testMethodName)

    def test_save_images_default(self):
        """ Test saving images with default settings. """
        results = save_images(
            self.jpeg_dataset, self.test_output_dir, image_format="jpg"
        )
        self.assertEqual(
            len(results),
            len(self.jpeg_data_dicts),
            "The number of results should be equal to the number of dataset samples.",
        )

        for result, label in zip(
            results, [d["category_codes"] for d in self.jpeg_data_dicts]
        ):
            file_path = result["path"]
            self.assertTrue(
                os.path.exists(file_path), f"The file {file_path} should exist."
            )
            self.assertEqual(
                result["label"], label, f"The label for {file_path} should be {label}."
            )

    def test_save_images_png_format(self):
        """ Test saving images in PNG format. """
        results = save_images(
            self.png_dataset, self.test_output_dir, image_format="png"
        )
        self.assertEqual(
            len(results),
            len(self.png_data_dicts),
            "The number of results should be equal to the number of dataset samples.",
        )

        for result, label in zip(
            results, [d["category_codes"] for d in self.png_data_dicts]
        ):
            file_path = result["path"]
            self.assertTrue(
                os.path.exists(file_path), f"The file {file_path} should exist."
            )
            self.assertTrue(
                file_path.endswith(".png"),
                f"The file {file_path} should have a .png extension.",
            )
            self.assertEqual(
                result["label"], label, f"The label for {file_path} should be {label}."
            )

    def test_save_images_custom_prefix_suffix(self):
        """ Test saving images with custom prefix and suffix. """
        prefix = "custom_prefix"
        suffix = "_suffix"
        results = save_images(
            self.jpeg_dataset,
            self.test_output_dir,
            image_format="jpg",
            prefix=prefix,
            suffix=suffix,
        )
        self.assertEqual(
            len(results),
            len(self.jpeg_data_dicts),
            "The number of results should be equal to the number of dataset samples.",
        )

        for result, label in zip(
            results, [d["category_codes"] for d in self.jpeg_data_dicts]
        ):
            file_path = result["path"]
            self.assertTrue(
                os.path.exists(file_path), f"The file {file_path} should exist."
            )
            self.assertTrue(
                file_path.startswith(os.path.join(self.test_output_dir, prefix)),
                f"The file {file_path} should start with {prefix}.",
            )
            self.assertTrue(
                file_path.endswith(suffix + ".jpg"),
                f"The file {file_path} should end with {suffix}.jpg.",
            )
            self.assertEqual(
                result["label"], label, f"The label for {file_path} should be {label}."
            )

    def test_save_images_sequential_naming(self):
        """ Test sequential naming of saved images. """
        results = save_images(
            self.jpeg_dataset, self.test_output_dir, image_format="jpg"
        )
        self.assertEqual(
            len(results),
            len(self.jpeg_data_dicts),
            "The number of results should be equal to the number of dataset samples.",
        )

        for i, result in enumerate(results):
            file_path = result["path"]
            unique_id = f"{i:04d}"
            expected_file_name = f"image_{unique_id}.jpg"
            self.assertTrue(
                file_path.endswith(expected_file_name),
                f"The file {file_path} should be named {expected_file_name}.",
            )
            self.assertEqual(
                result["label"],
                self.jpeg_data_dicts[i]["category_codes"],
                f"The label for {file_path} should be {self.jpeg_data_dicts[i]['category_codes']}.",
            )


if __name__ == "__main__":
    unittest.main()
