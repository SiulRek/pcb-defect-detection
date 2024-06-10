import warnings

import tensorflow as tf

from source.data_handling.helpers.label_manager import LabelManager
from source.data_handling.io.decode_image import decode_image


def create_dataset(data, label_type=None, category_names=None):
    """
    Creates a TensorFlow Dataset object from the given data containing file
    paths and labels using LabelManager for label encoding. Data can be a list
    of dictionaries or a pandas DataFrame.

    Args:
        - data (dicts, list of dicts or pandas.DataFrame): Data containing
            'path' and labels. 'path' should contain the relative file paths and
            labels should contain the corresponding labels for the specified
            'label_type'.
        - label_type (str, optional): Specifies the label encoding strategy
            ('binary', 'category_codes', 'sparse_category_codes', or
            'object_detection'). Default is None.
        - category_names (list, optional): The existing category names for
            label encoding if 'label_type' is not None. Default is None.

    Returns:
        - tf.data.Dataset: A TensorFlow Dataset containing tuples of (image,
            encoded label), where 'image' is the decoded image file and 'encoded
            label' is processed by LabelManager. If no labels are provided,
            returns a dataset of images only.
    """
    try:
        import pandas as pd

        pandas_installed = True
    except ImportError:
        pandas_installed = False

    if pandas_installed and isinstance(data, pd.DataFrame):
        paths = data["path"].tolist()
        labels = data["label"].tolist() if "label" in data.columns else None
    elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
        paths = [item["path"] for item in data]
        labels = [item.get("label") for item in data] if "label" in data[0] else None
    elif isinstance(data, dict):
        paths = data["path"]
        labels = data.get("label")
    else:
        msg = "Data must be a list of dictionaries, dictionary, or pandas"
        msg += "DataFrame."
        raise ValueError(msg)

    if label_type and labels:
        label_manager = LabelManager(label_type, category_names=category_names)
        labels = [label_manager.encode_label(label) for label in labels]
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        dataset = dataset.map(lambda path, label: (decode_image(path), label))
        return dataset
    if labels:
        msg = "No label type provided. Returning images only."
        warnings.warn(msg)
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.map(lambda path: decode_image(path))
    return dataset
