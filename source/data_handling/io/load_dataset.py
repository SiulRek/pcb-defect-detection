import os
import tensorflow as tf

from source.data_handling.io.load_and_decode_image import load_and_decode_image
from source.data_handling.helpers.label_manager import LabelManager


ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")


def load_dataset(data, label_type="category_codes", num_classes=None):
    """
    Creates a TensorFlow Dataset object from the given data containing file paths and labels
    using LabelManager for label encoding. Data can be a list of dictionaries or a pandas DataFrame.

    Args:
    - data (list of dicts or pandas.DataFrame): Data containing 'path' and 'category_codes'.
        'path' should contain the relative file paths and 'category_codes' should contain the
        corresponding labels for the specified 'label_type'.
    - label_type (str, optional): Specifies the label encoding strategy ('category_codes',
        'sparse_category_codes', or 'object_detection').
    - num_classes (int, optional): The number of classes for 'category_codes' label encoding.

    Returns:
        tf.data.Dataset: A TensorFlow Dataset containing tuples of (image, encoded label),
        where 'image' is the decoded image file and 'encoded label' is processed by LabelManager.
    """
    try:
        import pandas as pd

        pandas_installed = True
    except ImportError:
        pandas_installed = False

    label_manager = LabelManager(label_type, num_classes=num_classes)

    if pandas_installed and isinstance(data, pd.DataFrame):
        paths = data["path"].tolist()
        labels = [
            label_manager.get_label({"category_codes": sample["category_codes"]})
            for _, sample in data.iterrows()
        ]
    elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
        paths = [item["path"] for item in data]
        labels = [label_manager.get_label(item) for item in data]
    else:
        raise ValueError("Data must be a list of dictionaries or a pandas DataFrame.")

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(lambda path, label: (load_and_decode_image(path), label))

    return dataset
