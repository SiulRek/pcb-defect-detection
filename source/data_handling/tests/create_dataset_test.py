import unittest

import numpy as np
import pandas as pd
import tensorflow as tf

from source.data_handling.io.create_dataset import create_dataset
from source.testing.base_test_case import BaseTestCase


class TestCreateDataset(BaseTestCase):
    """ Test suite for the create_dataset function. """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.jpg_dict, cls.png_dict = cls.load_sign_language_digits_dict()

    def setUp(self):
        super().setUp()
        self.category_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    def _normalize_label(self, label):
        """ Helper function to normalize the label to a predefined format. """
        if isinstance(label, str):
            return int(label)
        if isinstance(label, list):
            return label.index(1)
        if isinstance(label, tf.Tensor):
            if label.shape[-1] == len(self.category_names):
                return tf.argmax(label).numpy()
            return label.numpy()
        raise ValueError(f"Invalid label type: {type(label)}")

    def _expected_one_hot_label(self, label):
        """ Helper function to get the expected one-hot label encoding as a numpy
        array. """
        one_hot_label = np.zeros(len(self.category_names))
        one_hot_label[int(label)] = 1
        return one_hot_label

    def _check_label(self, label, expected_label):
        """ Helper function to check the label against the expected label. """
        label = self._normalize_label(label)
        expected_label = self._normalize_label(expected_label)
        self.assertEqual(
            label, expected_label, f"Label mismatch {label} != {expected_label}"
        )

    def test_create_dataset_from_dicts_jpg(self):
        """ Test create_dataset with a list of dictionaries containing JPG images. """
        data = self.jpg_dict
        dataset = create_dataset(data, self.category_names)
        self.assertIsInstance(dataset, tf.data.Dataset)
        for i, (image, label) in enumerate(dataset):
            self.assertIsInstance(image, tf.Tensor)
            self.assertIsInstance(label, tf.Tensor)
            self._check_label(label, data["label"][i])

    def test_create_dataset_from_dicts_png(self):
        """ Test create_dataset with a list of dictionaries containing PNG images. """
        data = self.png_dict
        dataset = create_dataset(data, self.category_names)
        self.assertIsInstance(dataset, tf.data.Dataset)
        for i, (image, label) in enumerate(dataset):
            self.assertIsInstance(image, tf.Tensor)
            self.assertIsInstance(label, tf.Tensor)
            self._check_label(label, data["label"][i])

    def test_create_dataset_from_dataframe_jpg(self):
        """ Test create_dataset with a pandas DataFrame containing JPG images. """
        data = pd.DataFrame(self.jpg_dict)
        dataset = create_dataset(data, self.category_names)
        self.assertIsInstance(dataset, tf.data.Dataset)
        for i, (image, label) in enumerate(dataset):
            self.assertIsInstance(image, tf.Tensor)
            self.assertIsInstance(label, tf.Tensor)
            self._check_label(label, self.jpg_dict["label"][i])

    def test_create_dataset_from_dataframe_png(self):
        """ Test create_dataset with a pandas DataFrame containing PNG images. """
        data = pd.DataFrame(self.png_dict)
        dataset = create_dataset(data, self.category_names)
        self.assertIsInstance(dataset, tf.data.Dataset)
        for i, (image, label) in enumerate(dataset):
            self.assertIsInstance(image, tf.Tensor)
            self.assertIsInstance(label, tf.Tensor)
            self._check_label(label, self.png_dict["label"][i])

    def test_dataset_from_dicts(self):
        """ Test dataset creation from a list of dictionaries. """
        data = [
            {"path": self.png_dict["path"][0], "label": self.png_dict["label"][0]},
            {"path": self.jpg_dict["path"][1], "label": self.jpg_dict["label"][1]},
        ]
        dataset = create_dataset(data, self.category_names)
        self.assertIsInstance(dataset, tf.data.Dataset)
        for i, (image, label) in enumerate(dataset):
            self.assertIsInstance(image, tf.Tensor)
            self.assertIsInstance(label, tf.Tensor)
            self._check_label(label, data[i]["label"])

    def test_one_hot_encoding(self):
        """ Test one-hot encoding for category codes. """
        data = pd.DataFrame(self.jpg_dict)
        label_type = "category_codes"
        dataset = create_dataset(data, self.category_names, label_type=label_type)
        self.assertIsInstance(dataset, tf.data.Dataset)
        for i, (image, label) in enumerate(dataset):
            self.assertIsInstance(image, tf.Tensor)
            self.assertIsInstance(label, tf.Tensor)
            expected_label = self._expected_one_hot_label(self.jpg_dict["label"][i])
            self.assertTrue(np.array_equal(label.numpy(), expected_label))

    def test_invalid_data_type(self):
        """ Test if ValueError is raised for invalid data type. """
        data = "invalid_data_type"
        with self.assertRaises(ValueError):
            create_dataset(data, self.category_names)


if __name__ == "__main__":
    unittest.main()
