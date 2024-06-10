import math
import os

import tensorflow as tf


def save_images(
    dataset, output_dir, image_format="jpg", prefix="image", start_number=0
):
    """
    Saves images from dataset to the specified directory. Unlabeled dataset is
    allowed.

    Args:
        - dataset (tf.data.Dataset): The dataset containing images and
            labels.
        - output_dir (str): The directory to save the encoded image files.
        - image_format (str, optional): The format for saving images ('jpg'
            or 'png').
        - prefix (str or function, optional): The prefix for naming the
            saved images. If a function is provided, it should take a label and
            return a string.
        - start_number (int, optional): The starting number for the
            sequential naming.
    """
    os.makedirs(output_dir, exist_ok=True)

    labeled = isinstance(dataset.element_spec, tuple)
    sample_image = next(iter(dataset))[0] if labeled else next(iter(dataset))

    if sample_image.dtype in [tf.float16, tf.float32, tf.float64]:
        dataset = dataset.map(
            lambda x, y=None: (
                (tf.cast(x * 255, tf.uint8), y)
                if labeled
                else tf.cast(x * 255, tf.uint8)
            )
        )
    elif sample_image.dtype != tf.uint8:
        msg = "Image data type not supported."
        raise ValueError(msg)

    if sample_image.shape.ndims == 4:
        dataset = dataset.unbatch()

    if image_format not in ["jpg", "png"]:
        msg = "Image format not supported. Use 'jpg' or 'png'."
        raise ValueError(msg)

    num_samples = sum(1 for _ in dataset)
    num_digits = max(4, math.ceil(math.log10(num_samples + start_number)))

    encode_fn = tf.image.encode_jpeg if image_format == "jpg" else tf.image.encode_png

    for seq_number, sample in enumerate(dataset, start=start_number):
        image, label = sample if labeled else (sample, None)
        prefix_str = prefix(label) if callable(prefix) else prefix

        file_name = f"{prefix_str}_{seq_number:0{num_digits}d}.{image_format}"
        file_path = os.path.join(output_dir, file_name)

        encoded_image = encode_fn(image)
        tf.io.write_file(file_path, encoded_image)
