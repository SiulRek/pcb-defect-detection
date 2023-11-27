import tensorflow as tf


def correct_tf_image_shape(tf_image):
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
    mean = tf.reduce_mean(tensor)
    squared_deviations = tf.square(tensor - mean)
    variance = tf.reduce_mean(squared_deviations)
    std_dev = tf.sqrt(variance)
    return std_dev 
