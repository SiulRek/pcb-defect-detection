import os
import shutil
import unittest

import tensorflow as tf

from source.data_handling.io.save_images import save_images
from source.testing.base_test_case import BaseTestCase


class TestSaveImages(BaseTestCase):
    """ Test suite for the save_images function. """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def setUp(self):
        super().setUp()
        self.dataset = self.load_sign_language_digits_dataset(
            sample_num=5, labeled=True
        )

    def test_saving_images_default_settings(self):
        """ Test saving images with default settings. """
        results_dir = os.path.join(self.temp_dir, "saving_images_default_settings")
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir)
        save_images(self.dataset, results_dir)
        list_dir = os.listdir(results_dir)
        for file in list_dir:
            self.assertTrue(
                file.endswith(".jpg"), "Image should be saved in default JPG format."
            )
        self.assertEqual(
            len(list_dir), len(list(self.dataset)), "All images should be saved."
        )

    def test_saving_images_png_format(self):
        """ Test saving images in PNG format. """
        results_dir = os.path.join(self.temp_dir, "saving_images_png_format")
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir)
        save_images(self.dataset, results_dir, image_format="png")
        list_dir = os.listdir(results_dir)
        for file in list_dir:
            self.assertTrue(
                file.endswith(".png"), "Image should be saved in PNG format."
            )
        self.assertEqual(
            len(list_dir), len(list(self.dataset)), "All images should be saved."
        )

    def test_saving_images_jpg_format(self):
        """ Test saving images in JPG format. """
        results_dir = os.path.join(self.temp_dir, "saving_images_jpg_format")
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir)
        save_images(self.dataset, results_dir, image_format="jpg")
        list_dir = os.listdir(results_dir)
        for file in list_dir:
            self.assertTrue(
                file.endswith(".jpg"), "Image should be saved in JPG format."
            )
        self.assertEqual(
            len(list_dir), len(list(self.dataset)), "All images should be saved."
        )

    def test_saving_images_with_string_prefix(self):
        """ Test saving images with a string prefix. """
        results_dir = os.path.join(self.temp_dir, "saving_images_with_string_prefix")
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir)
        save_images(self.dataset, results_dir, prefix="test_prefix")
        list_dir = os.listdir(results_dir)
        for file in list_dir:
            self.assertTrue(
                "test_prefix" in file,
                "Image filename should contain the string prefix.",
            )
        self.assertEqual(
            len(list_dir), len(list(self.dataset)), "All images should be saved."
        )

    def test_saving_images_with_function_prefix(self):
        """ Test saving images with a function prefix. """

        def prefix_function(label):
            label = tf.argmax(label)
            return f"label_{label.numpy()}"

        results_dir = os.path.join(self.temp_dir, "saving_images_with_function_prefix")
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir)
        save_images(self.dataset, results_dir, prefix=prefix_function)
        list_dir = os.listdir(results_dir)
        for file in list_dir:
            self.assertTrue(
                file.startswith("label_"),
                "Image filename should contain the function prefix.",
            )
        self.assertEqual(
            len(list_dir), len(list(self.dataset)), "All images should be saved."
        )

    def test_sequential_naming_of_saved_images(self):
        """ Test sequential naming of saved images. """
        results_dir = os.path.join(self.temp_dir, "sequential_naming_of_saved_images")
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir)
        save_images(self.dataset, results_dir, start_number=10)
        list_dir = os.listdir(results_dir)
        list_dir.sort()
        for i, file in enumerate(list_dir):
            expected_name = f"image_{i+10:04d}.jpg"
            self.assertTrue(
                file.endswith(expected_name),
                "Image filename should be sequentially named.",
            )
        self.assertEqual(
            len(list_dir), len(list(self.dataset)), "All images should be saved."
        )

    def test_saving_floating_point_images(self):
        """ Test saving images with floating point data type. """
        results_dir = os.path.join(self.temp_dir, "saving_floating_point_images")
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir)
        dataset = self.dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
        save_images(dataset, results_dir)
        list_dir = os.listdir(results_dir)
        for file in list_dir:
            self.assertTrue(
                file.endswith(".jpg"), "Image should be saved in JPG format."
            )
            self.assertTrue(
                os.path.exists(os.path.join(results_dir, file)),
                "Image file should exist.",
            )
        self.assertEqual(
            len(list_dir), len(list(self.dataset)), "All images should be saved."
        )

    def test_saving_images_with_batched_dataset(self):
        """ Test saving images with a batched dataset. """
        results_dir = os.path.join(self.temp_dir, "saving_images_with_batched_dataset")
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir)
        dataset = self.dataset.batch(2)
        save_images(dataset, results_dir)
        list_dir = os.listdir(results_dir)
        for file in list_dir:
            self.assertTrue(
                file.endswith(".jpg"), "Image should be saved in JPG format."
            )
            self.assertTrue(
                os.path.exists(os.path.join(results_dir, file)),
                "Image file should exist.",
            )
        self.assertEqual(
            len(list_dir), len(list(self.dataset)), "All images should be saved."
        )

if __name__ == "__main__":
    unittest.main()
