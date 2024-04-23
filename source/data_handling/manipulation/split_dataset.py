import tensorflow as tf

def split_dataset(dataset, train_size=0.8, val_size=0.1, test_size=0.1):
    """
    Splits a TensorFlow dataset into training, validation, and test sets based on the specified
    proportions.

    Args:
    - dataset (tf.data.Dataset): The TensorFlow dataset to split.
    - train_size (float, optional): Proportion of the dataset to include in the training set.
    - val_size (float, optional): Proportion of the dataset to include in the validation set.
    - test_size (float, optional): Proportion of the dataset to include in the test set.

    Returns:
    - tf.data.Dataset: The training dataset.
    - tf.data.Dataset: The validation dataset.
    - tf.data.Dataset: The test dataset.
    """
    if train_size + val_size + test_size != 1.0:
        raise ValueError("The sum of train_size, val_size, and test_size should be 1.0.")

    dataset_size = dataset.cardinality().numpy()
    train_size = int(train_size * dataset_size)
    val_size = int(val_size * dataset_size)

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size + val_size)

    return train_dataset, val_dataset, test_dataset