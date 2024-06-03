import os

import tensorflow as tf


def _bytes_feature(value):
    """ Returns a bytes_list from a string / byte. """
    if isinstance(value, type(tf.constant(0))):  # if value is a Tensor
        value = value.numpy()  # get its numpy value
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_sample_for_png(image, label):
    """ Serializes an image and label pair to a tf.train.Example proto. The image is
    expected to be a PNG image. """
    feature = {
        "image": _bytes_feature(tf.io.encode_png(image).numpy()),
        "label": _bytes_feature(tf.io.serialize_tensor(label).numpy()),
    }
    sample_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return sample_proto.SerializeToString()


def serialize_sample_for_jpeg(image, label):
    """ Serializes an image and label pair to a tf.train.Example proto. The image is
    expected to be a JPEG image. """
    feature = {
        "image": _bytes_feature(tf.io.encode_jpeg(image).numpy()),
        "label": _bytes_feature(tf.io.serialize_tensor(label).numpy()),
    }
    sample_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return sample_proto.SerializeToString()


def serialize_dataset_to_tf_record(dataset, filepath, image_format):
    """
    Saves a dataset to a TFRecord file.

    Args:
        - dataset (tf.data.Dataset): A dataset object containing image and
            label pairs.
        - filepath (str): The path to the output TFRecord file.
        - image_format (str): The format of the images in the dataset. It
            can be either 'png' or 'jpeg'.
    """
    if image_format == "png":
        serialize_sample = serialize_sample_for_png
    elif image_format == "jpeg":
        serialize_sample = serialize_sample_for_jpeg
    else:
        msg = "Image format '{}' is not supported.".format(image_format)
        msg += " Please use either 'png' or 'jpeg'."
        raise ValueError(msg)

    if not os.path.exists(os.path.dirname(filepath)):
        msg = f"Directory '{os.path.dirname(filepath)}' does not exist."
        raise FileNotFoundError(msg)
    
    with tf.io.TFRecordWriter(filepath) as writer:
        for image, label in dataset:
            example = serialize_sample(image, label)
            writer.write(example)
