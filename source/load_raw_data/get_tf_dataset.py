import os
import random

import tensorflow as tf

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')


def get_tf_dataset_from_df(dataframe, random_seed=34, sample_num=-1):
    """
    Creates a TensorFlow Dataset object from the given dataframe containing file paths and category codes.
    
    Parameters:
    - dataframe (pandas.DataFrame): A dataframe containing columns 'path' and 'category_codes'.
        'path' should contain the relative file paths and 'category_codes' should contain the corresponding category codes.
    - random_seed (int, optional): The random seed for shuffling the dataset. Defaults to 34.
    - sample_num (int, optional): Numbers of samples to take from the dataframe. Defaults to -1 -> All Samples are taken.
    
    Returns:
    - tf.data.Dataset: A TensorFlow Dataset object containing shuffled paths and corresponding targets.
    """

    paths = []
    targets = []
    for _, sample in dataframe.iterrows():
        if sample['path'] not in paths:
            paths.append(os.path.join(ROOT_DIR, sample['path']))
            targets.append(tf.constant(sample['category_codes'], dtype=tf.int8))
    
    if sample_num == -1:
        indices = list(range(len(paths)))
    elif sample_num <= len(paths):
        indices = list(range(sample_num))
    else:
        raise ValueError(f'Sample Number {sample_num} is outside the range.')

    random.seed(random_seed) 
    random.shuffle(indices)
    shuffled_targets = [targets[i] for i in indices]
    shuffled_paths = [paths[i] for i in indices]

    image_dataset = tf.data.Dataset.from_tensor_slices((shuffled_paths, shuffled_targets))
    image_dataset = image_dataset.map(load_and_decode_image)
    image_dataset = image_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) # Allows parallel processing of multiple items.
    return image_dataset

def load_and_decode_image(*image_data):
    image = tf.io.read_file(image_data[0])
    image = tf.image.decode_jpeg(image, channels=3)
    return (image, image_data[1])
