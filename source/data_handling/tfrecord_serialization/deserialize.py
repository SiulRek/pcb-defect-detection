import os

import tensorflow as tf


def parse_tfrecord(sample_proto):
    """Parses a serialized Example proto. It parses the image and label tensors from the serialized 
    data."""
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.string),
    }
    sample = tf.io.parse_single_example(sample_proto, feature_description)
    image = tf.io.decode_image(sample["image"], channels=3)
    label = tf.io.parse_tensor(sample["label"], out_type=tf.int8)
    return image, label


def deserialize_dataset_from_tfrecord(filepath):
    """Loads a TFRecord file into a tf.data.Dataset object."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"tfrecord '{filepath}' does not exist.")

    raw_dataset = tf.data.TFRecordDataset(filepath)
    parsed_dataset = raw_dataset.map(parse_tfrecord)
    return parsed_dataset
