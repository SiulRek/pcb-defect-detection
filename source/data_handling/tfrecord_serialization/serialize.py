import os

import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_sample(image, label):
    """Serializes an image-label pair into a TFRecord-compatible Example object."""
    feature = {
        "image": _bytes_feature(tf.io.encode_png(image).numpy()),
        "label": _bytes_feature(tf.io.serialize_tensor(label).numpy()),
    }
    sample_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return sample_proto.SerializeToString()


def serialize_dataset_to_tf_record(dataset, filepath):
    """Saves a tf.data.Dataset object to a TFRecord file."""
    if not os.path.exists(os.path.dirname(filepath)):
        raise FileNotFoundError(
            f"Directory '{os.path.dirname(filepath)}' does not exist."
        )

    with tf.io.TFRecordWriter(filepath) as writer:
        for image, label in dataset:
            example = serialize_sample(image, label)
            writer.write(example)
