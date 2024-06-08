import unittest

import tensorflow as tf

from source.data_handling.manipulation.split_dataset import split_dataset
from source.testing.base_test_case import BaseTestCase


class TestDatasetFunctions(BaseTestCase):
    """ Test suite for the split_dataset function. """

    def setUp(self):
        super().setUp()
        self.dataset = tf.data.Dataset.range(100)

    def test_split_proportions(self):
        train, val, test = split_dataset(self.dataset)
        self.assertEqual(
            train.cardinality().numpy(), 80, "Training set size should be 80% of total."
        )
        self.assertEqual(
            val.cardinality().numpy(), 10, "Validation set size should be 10% of total."
        )
        self.assertEqual(
            test.cardinality().numpy(), 10, "Test set size should be 10% of total."
        )

    def test_split_proportions_error(self):
        with self.assertRaises(ValueError):
            split_dataset(self.dataset, train_size=0.7, val_size=0.2, test_size=0.2)


if __name__ == "__main__":
    unittest.main()
