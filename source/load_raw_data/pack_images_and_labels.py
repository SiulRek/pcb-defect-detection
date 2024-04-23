import tensorflow as tf


def pack_images_and_labels(image_dataset, label_dataset):
    """Packs two tf.data.Dataset objects for images and labels into a single tf.data.Dataset.

    Args:
    - image_dataset (tf.data.Dataset): A dataset of image tensors.
    - label_dataset (tf.data.Dataset): A dataset of label tensors.

    Returns:
    - tf_dataset (tf.data.Dataset): A tf.data.Dataset object containing tuples of (image, label).
    """
    return tf.data.Dataset.zip((image_dataset, label_dataset))
