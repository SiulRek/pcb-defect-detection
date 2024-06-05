# NOTE: The functionality in DataHandling is not used in the module
# here, as this makes the testing framework more independent.
import os

import tensorflow as tf


def _bytes_feature(value):
    """ Returns a bytes_list from a string / byte. """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_image(image_path):
    """
    Reads an image from a file, decodes it, and serializes it into a
    tf.train.Example.

    Args:
        - image_path (str): Path to the image file.

    Returns:
        - tf.train.Example: The Example proto containing the image.
    """
    image_string = tf.io.read_file(image_path)
    image_decoded = tf.image.decode_png(image_string)
    image_resized = tf.image.resize(image_decoded, [224, 224])
    image_bytes = tf.io.encode_png(tf.cast(image_resized, tf.uint8))
    feature = {"image_raw": _bytes_feature(image_bytes)}
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def create_tfrecord_from_images(image_directory, output_filepath):
    """
    Creates a TFRecord file from images in a specified directory.

    Args:
        - image_directory (str): The directory where image files are
            located.
        - output_filepath (str): The path to store the TFRecord file.
    """
    with tf.io.TFRecordWriter(output_filepath) as writer:
        for filename in os.listdir(image_directory):
            if filename.lower().endswith((".png")):
                image_path = os.path.join(image_directory, filename)
                example = serialize_image(image_path)
                writer.write(example)

    print(f"TFRecord file has been created at {output_filepath}")


if __name__ == "__main__":
    image_directory = "source/testing/image_data/geometrical_forms"
    output_filepath = "source/testing/image_data/tf_records/geometrical_forms.tfrecord"
    create_tfrecord_from_images(image_directory, output_filepath)
