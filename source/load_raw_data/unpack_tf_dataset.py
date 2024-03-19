def unpack_tf_dataset(tf_dataset):
    """ Unpacks a tf.data.Dataset into two tf.data.Dataset objects for images and labels.

    Args: 
    - tf_dataset (tf.data.Dataset): A tf.data.Dataset object containing tuples of (image, label).
        'image' is the decoded image file and 'label' is an integer label.
    
    Returns:
    - image_dataset (tf.data.Dataset): A dataset of image tensors.
    - label_dataset (tf.data.Dataset): A dataset of label tensors.
    """
    image_dataset = tf_dataset.map(lambda image, label: image)
    label_dataset = tf_dataset.map(lambda image, label: label)
    return image_dataset, label_dataset
