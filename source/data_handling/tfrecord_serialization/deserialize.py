import os

import tensorflow as tf


def parse_tfrecord(sample_proto, label_dtype=None):
    """ Parses a serialized Example proto to parse the image and optionally label
    tensors. """
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
    }
    if label_dtype is not None:
        feature_description["label"] = tf.io.FixedLenFeature([], tf.string)

    sample = tf.io.parse_single_example(sample_proto, feature_description)
    image = tf.io.decode_image(sample["image"], channels=3)

    if "label" in sample:
        label = tf.io.parse_tensor(sample["label"], out_type=label_dtype)
        return image, label
    else:
        return image


def deserialize_dataset_from_tfrecord(filepath, label_dtype=None):
    """
    Loads a TFRecord file into a tf.data.Dataset object.

    Args:
        - filepath (str): The path to the TFRecord file.
        - label_dtype (tf.DType, optional): The data type of the labels in
            the dataset. If None, assumes there are no labels.

    Returns:
        - tf.data.Dataset: A dataset object containing image and optionally
            label pairs.
    """
    if not os.path.exists(filepath):
        msg = f"tfrecord '{filepath}' does not exist."
        raise FileNotFoundError(msg)

    raw_dataset = tf.data.TFRecordDataset(filepath)
    parsed_dataset = raw_dataset.map(lambda x: parse_tfrecord(x, label_dtype))
    return parsed_dataset
