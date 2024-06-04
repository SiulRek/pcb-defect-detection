import os

import tensorflow as tf

from source.data_handling.helpers.label_manager import LabelManager
from source.data_handling.io.decode_image import decode_image

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")


def create_dataset(data, category_names, label_type="category_codes"):
    """
    Creates a TensorFlow Dataset object from the given data containing file
    paths and labels using LabelManager for label encoding. Data can be a list
    of dictionaries or a pandas DataFrame.

    Args:
        - data (dicts, list of dicts or pandas.DataFrame): Data containing 'path'
            and labels. 'path' should contain the relative file paths and labels
            should contain the corresponding labels for the specified
            'label_type'.
        - category_names (list): The existing category names for label
            encoding.
        - label_type (str, optional): Specifies the label encoding strategy
            ('binary', 'category_codes', 'sparse_category_codes', or
            'object_detection').

    Returns:
        - tf.data.Dataset: A TensorFlow Dataset containing tuples of (image,
            encoded label), where 'image' is the decoded image file and 'encoded
            label' is processed by LabelManager.
    """
    try:
        import pandas as pd

        pandas_installed = True
    except ImportError:
        pandas_installed = False
    label_manager = LabelManager(label_type, category_names=category_names)

    if pandas_installed and isinstance(data, pd.DataFrame):
        paths = data["path"].tolist()
        labels = [label_manager.encode_label(label) for label in data["label"]]
    elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
        paths = [item["path"] for item in data]
        labels = [label_manager.encode_label(item["label"]) for item in data]
    elif isinstance(data, dict):
        paths = data["path"]
        labels = [label_manager.encode_label(label) for label in data["label"]]
    else:
        msg = "Data must be a list of dictionaries or a pandas DataFrame."
        raise ValueError(msg)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(lambda path, label: (decode_image(path), label))

    return dataset
