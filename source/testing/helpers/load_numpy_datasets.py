import os

import numpy as np
import tensorflow as tf

ROOT_DIR = os.path.join(os.path.abspath(__file__), "..", "..", "..", "..")
NP_DATASET_DIR = os.path.join(ROOT_DIR, "testing", "image_data", "numpy_datasets")


def load_sign_digits_dataset(sample_num=None, labeled=True):
    """ Load the sign language digits dataset used for testing. 

    Args:
        - sample_num (int): The number of samples to load from the dataset.
        - labeled (bool): Whether to return the dataset with labels.
    
    Returns:
        - tf.data.Dataset: The sign language digits dataset to be used for testing.
    """
    X = np.load(os.path.join(NP_DATASET_DIR, "sign_language_digits_X.npy"))
    Y = np.load(os.path.join(NP_DATASET_DIR, "sign_language_digits_Y.npy"))
    if sample_num:
        X = X[:sample_num]
        Y = Y[:sample_num]
    if labeled:
        return tf.data.Dataset.from_tensor_slices((X, Y))
    else:
        return tf.data.Dataset.from_tensor_slices(X)
