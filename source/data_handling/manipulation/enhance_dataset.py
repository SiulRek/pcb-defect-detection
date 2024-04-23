import tensorflow as tf


def enhance_dataset(
    dataset,
    batch_size=None,
    shuffle=False,
    random_seed=None,
    prefetch_buffer_size=tf.data.experimental.AUTOTUNE,
    cache=False,
    repeat_num=None,
):
    """
    Enhances a TensorFlow dataset by applying shuffling, batching, prefetching, and optionally
    repeating a specified number of times or indefinitely if repeat_num is None and the repeat
    parameter is set to True.

    Args:
    - dataset (tf.data.Dataset): The initial TensorFlow dataset to enhance.
    - batch_size (int, optional): Size of batches of data.
    - shuffle (bool, optional): Whether to shuffle the dataset. Default is False.
    - random_seed (int, optional): Seed for random shuffling if shuffle is True.
    - prefetch_buffer_size (int, optional): Number of batches to prefetch (default is
        tf.data.experimental.AUTOTUNE).
    - cache (bool, optional): Whether to cache the dataset. Default is False.
    - repeat_num (int, optional): Number of times to repeat the dataset. None for indefinite repeat.

    Returns:
    - tf.data.Dataset: The enhanced TensorFlow dataset.
    """
    if shuffle:
        shuffle_buffer_size = dataset.cardinality().numpy()
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=random_seed)

    if batch_size:
        dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

    if repeat_num is not None:
        dataset = dataset.repeat(repeat_num)

    if cache:
        dataset = dataset.cache()

    return dataset
