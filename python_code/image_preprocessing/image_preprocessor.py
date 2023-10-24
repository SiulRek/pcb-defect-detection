import tensorflow as tf

#IMPORTANT ANNOTATIONS: The current class implementation serves jusst as a template and as documentation of the current software plan.


class ImagePreprocessor:
    def __init__(self, preprocessing_config=None):
        """Initializes the ImagePreprocessor with a given configuration.

        Args:
            preprocessing_config (dict, optional): Configuration dict specifying which preprocessing steps to apply. 
                                                   Each key is a step name, and the associated value is a dictionary of arguments for that step.
        """
        self.config = preprocessing_config or {}

    def process(self, images):
        """Processes the images according to the provided configuration.
        
        Args:
            images (tf.Tensor): Input images to preprocess.
        
        Returns:
            tf.Tensor: Preprocessed images.
        """
        if "resize" in self.config:
            images = self.resize_images(images, **self.config["resize"])
        
        if "enhance_contrast" in self.config:
            images = self.enhance_contrast(images, **self.config["enhance_contrast"])
        
        if "reduce_noise" in self.config:
            images = self.reduce_noise(images, **self.config["reduce_noise"])
        
        # other steps can be added here similarly

        return images

    def resize_images(self, images, width, height):
        """Resizes the images to the specified dimensions.

        Args:
            images (tf.Tensor): Input images.
            width (int): Desired width.
            height (int): Desired height.

        Returns:
            tf.Tensor: Resized images.
        """
        return tf.image.resize(images, [height, width])

    def enhance_contrast(self, images):
        """Enhances the contrast of the images.

        """
        pass

        

    def reduce_noise(self, images, method='gaussian'):
        """Reduces noise in the images.
        """
        pass


    # Add other processing functions similarly...
