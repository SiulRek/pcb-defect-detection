import tensorflow as tf


def correct_image_tensor_shape(tf_image):
    """
    Corrects the shape of a TensorFlow image tensor based on the inferred dimensions.
    
    Parameters:
    - tf_image (tf.Tensor): The input image tensor.
    
    Returns:
    - tf.Tensor: A reshaped tensor based on inferred dimensions.
    """
    
    dims = tf.shape(tf_image)
    height = dims[0]
    width = dims[1]

    # Check if the image is grayscale (2D) and if so, reshape it to 3D with 1 channel.
    if len(dims) == 2:
        reshaped_image = tf.reshape(tf_image, [height, width, 1])
    else:
        channel_num = dims[2]
        reshaped_image = tf.reshape(tf_image, [height, width, channel_num])
    
    return reshaped_image   

def reduce_std(tensor):
    """
    Computes the standard deviation of a tensor.

    Parameters:
    tensor (tf.Tensor): The input tensor.

    Returns:
    tf.Tensor: The standard deviation of elements in the tensor as a scalar.
    """
    mean = tf.reduce_mean(tensor)
    squared_deviations = tf.square(tensor - mean)
    variance = tf.reduce_mean(squared_deviations)
    std_dev = tf.sqrt(variance)
    return std_dev 

import tensorflow as tf

def squared_difference(tensor_a, tensor_b):
    """
    Compute the squared difference of a tensor and a scalar or another tensor.

    Parameters:
    tensor_a (Tensor): A `Tensor`.
    tensor_b (scalar or Tensor): A scalar or a `Tensor` with the same type and shape as `tensor_a`.

    Returns:
    Tensor: A `Tensor` containing the squared difference of the input tensor and the scalar or tensor.
    """
    difference = tensor_a - tensor_b  # Broadcasting happens here if tensor_b is a scalar
    squared_diff = tf.square(difference)
    return squared_diff
