import math
import os

import tensorflow as tf


def save_images(
    dataset, output_dir, image_format="jpg", prefix="image", start_number=0
):
    """
    Saves a dataset of images and labels to file paths.

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

    Returns:
        - list of dict: A list of dictionaries with 'path' and 'label' keys.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if image_format not in ["jpg", "png"]:
        msg = "Image format not supported. Use 'jpg' or 'png'."
        raise ValueError(msg)

    num_samples = sum(1 for _ in dataset)
    num_digits = max(4, math.ceil(math.log10(num_samples + start_number)))

    results = []
    seq_number = start_number

    if image_format == "jpg":
        encode_fn = tf.image.encode_jpeg
    else:
        encode_fn = tf.image.encode_png

    for image, label in dataset:
        unique_id = f"{seq_number:0{num_digits}d}"
        seq_number += 1

        if callable(prefix):
            prefix_str = prefix(label)
        else:
            prefix_str = prefix

        file_name = f"{prefix_str}_{unique_id}.{image_format}"
        file_path = os.path.join(output_dir, file_name)

        encoded_image = encode_fn(image)
        tf.io.write_file(file_path, encoded_image)

        results.append({"path": file_path, "label": label.numpy()})

    return results
