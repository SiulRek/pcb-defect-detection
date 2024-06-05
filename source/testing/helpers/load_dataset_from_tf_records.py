import tensorflow as tf


def _parse_image_function(proto):
    """
    Parses an image from the tf.train.Example.

    Args:
        - proto (tf.Tensor): A scalar string tensor, a single serialized
            Example.

    Returns:
        - tf.Tensor: A tensor representing the parsed image.
    """
    image_feature_description = {
        "image_raw": tf.io.FixedLenFeature([], tf.string),
    }
    parsed_features = tf.io.parse_single_example(proto, image_feature_description)
    image = tf.io.decode_png(parsed_features["image_raw"], channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def load_dataset_from_tf_records(tf_records_path):
    """
    Load and parse the dataset from a TFRecord file.

    Args:
        - tf_records_path (str): The file path to the TFRecord file.

    Returns:
        - tf.data.Dataset: The loaded and parsed dataset.
    """
    raw_dataset = tf.data.TFRecordDataset(tf_records_path)
    parsed_dataset = raw_dataset.map(_parse_image_function)
    return parsed_dataset
