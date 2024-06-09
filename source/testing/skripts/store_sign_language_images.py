import os

from PIL import Image
import numpy as np


def save_sign_language_digits_images(data_dir, output_dir, sample_num=5):
    """
    Save a specified number of images from the sign language digits dataset to
    the given directory with the naming convention
    image_<index>_sign_<label>.png.

    Args:
        - data_dir (str): Directory where the numpy dataset is located.
        - output_dir (str): Directory to save the images.
        - sample_num (int): Number of images to save. Defaults to 5.
    """
    dataset_dir = os.path.join(data_dir, "numpy_datasets", "sign_language_digits")
    X = np.load(os.path.join(dataset_dir, "X.npy"))
    Y = np.load(os.path.join(dataset_dir, "Y.npy"))

    X = (X * 255).astype(np.uint8)
    if X.ndim == 3:
        X = np.stack([X] * 3, axis=-1)

    # Shuffle
    random_indices = np.random.permutation(len(X))
    X = X[random_indices]
    Y = Y[random_indices]
    if sample_num:
        X = X[:sample_num]
        Y = Y[:sample_num]

    os.makedirs(output_dir, exist_ok=True)

    for i, (image, label) in enumerate(zip(X, Y)):
        image_pil = Image.fromarray(image)
        label = np.argmax(label)
        image_pil.save(os.path.join(output_dir, f"image_{i + 6}_sign_{label}.jpg"))


if __name__ == "__main__":
    data_dir = "./source/testing/image_data"
    output_dir = "./source/testing/image_data/sign_language_digits"
    sample_num = 5

    save_sign_language_digits_images(data_dir, output_dir, sample_num)
