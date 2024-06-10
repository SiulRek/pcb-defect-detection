import os

import tensorflow as tf


def _bytes_feature(value):
    """ Returns a bytes_list from a string / byte. """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_sample_for_png(image, label=None):
    """ Serializes an image and optionally a label pair to a tf.train.Example proto.
    The image is expected to be a PNG image. """
    feature = {"image": _bytes_feature(tf.io.encode_png(image).numpy())}
    if label is not None:
        feature["label"] = _bytes_feature(tf.io.serialize_tensor(label).numpy())

    sample_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return sample_proto.SerializeToString()


def serialize_sample_for_jpeg(image, label=None):
    """ Serializes an image and optionally a label pair to a tf.train.Example proto.
    The image is expected to be a JPEG image. """
    feature = {"image": _bytes_feature(tf.io.encode_jpeg(image).numpy())}
    if label is not None:
        feature["label"] = _bytes_feature(tf.io.serialize_tensor(label).numpy())

    sample_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return sample_proto.SerializeToString()


def serialize_dataset_to_tf_record(dataset, filepath, image_format):
    """
    Saves a dataset to a TFRecord file. If the dataset is batched, it will be
    unbatched before saving. Labeled dataset is allowed and will be serialized
    as image and label pairs.

    Args:
        - dataset (tf.data.Dataset): A dataset object containing image and
            optionally label pairs.
        - filepath (str): The path to the output TFRecord file.
        - image_format (str): The format of the images in the dataset. It
            can be either 'png' or 'jpeg'.
    """
    if image_format == "png":
        serialize_sample = serialize_sample_for_png
    elif image_format == "jpeg":
        serialize_sample = serialize_sample_for_jpeg
    else:
        msg = f"Image format '{image_format}' is not supported. Please use"
        msg += "either 'png' or 'jpeg'."
        raise ValueError(msg)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    element_spec = dataset.element_spec
    labeled = isinstance(element_spec, tuple)
    if (labeled and element_spec[0].shape.ndims == 4) or (
        not labeled and element_spec.shape.ndims == 4
    ):
        dataset = dataset.unbatch()

    with tf.io.TFRecordWriter(filepath) as writer:
        for sample in dataset:
            if labeled:
                image, label = sample
            else:
                image, label = sample, None
            example = serialize_sample(image, label)
            writer.write(example)
