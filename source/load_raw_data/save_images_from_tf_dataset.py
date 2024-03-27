import tensorflow as tf
import os
from PIL import Image
import numpy as np


def save_images_from_tf_dataset(tf_dataset, directory):
    """     
    Save images from a tf.data.Dataset object to a directory.

    Args:
        tf_dataset (tf.data.Dataset): A tf.data.Dataset object containing images. Format: (image, label).
        directory (str): The directory where the images will be saved.
    """

    if not os.path.exists(directory):
        os.makedirs(directory)

    for i, (image_tensor, _) in enumerate(tf_dataset):
        image = image_tensor.numpy()

        if image.dtype == np.float32 or image.dtype == np.float64 or image.dtype == np.float16:
            image = (image * 255).astype(np.uint8)
        elif image.dtype == np.uint8:
            pass
        else:
            raise ValueError(f"Unsupported image data type: {image.dtype}")

        if image.shape[2] == 1:
            image = Image.fromarray(image.squeeze(-1), mode='L')
        elif image.shape[2] == 3 and image.shape[2] == 3:
            image = Image.fromarray(image, mode='RGB')
        else:
            raise ValueError("Unsupported image format")

        file_path = os.path.join(directory, f'image_{i}.png')
        image.save(file_path)

