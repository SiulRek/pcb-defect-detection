import tensorflow as tf


def decode_image(image_path):
    """
    Dynamically decodes an image from a given path based on its extension.
    Supports JPEG and PNG formats. Uses TensorFlow operations to handle path
    operations directly on tensors.

    Args:
        - image_path (str): The path to the image file.
    
    Returns:
        - tf.Tensor: A tensor representing the image.
    """
    image = tf.io.read_file(image_path)

    def decode_jpeg(image):
        return tf.image.decode_jpeg(image, channels=3)

    def decode_png(image):
        return tf.image.decode_png(image, channels=3)

    is_jpeg = tf.strings.regex_full_match(image_path, r".*\.(jpeg|jpg)$")
    is_png = tf.strings.regex_full_match(image_path, r".*\.png$")

    image = tf.cond(
        is_jpeg,
        lambda: decode_jpeg(image),
        lambda: tf.cond(
            is_png,
            lambda: decode_png(image),
            lambda: tf.cast(tf.constant([]), tf.uint8),
        ),
    )
    return image