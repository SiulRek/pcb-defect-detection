import tensorflow as tf

#IMPORTANT ANNOTATIONS: The current class implementation serves jusst as a template and as documentation of the current software plan.


class ImagePreprocessor:
    def __init__(self, pipeline=None):
        """Initializes the ImagePreprocessor with a given pipeline.

        Args:
            pipeline (dict, optional): Pipeline represented as configuartion dict specifying which preprocessing steps to apply. 
                                                   Each key is a step name, and the associated value is a dictionary of arguments for that step.
        """
        self.pipeline = pipeline or {}

    def process(self, image_dataset):
        """Processes the images according to the provided configuration.
        
        Args:
            images (tf.Tensor): Input images to preprocess.
        
        Returns:
            tf.Tensor: Preprocessed images.
        """

        processed_dataset = image_dataset
        
        for key in self.pipeline.keys:

            if key == "resize":
                processed_dataset = self.resize_images(processed_dataset, **self.pipeline["resize"])
            
            if key == "enhance_contrast":
                processed_dataset = self.enhance_contrast(processed_dataset, **self.pipeline["enhance_contrast"])
            
            if key == "reduce_noise":
                processed_dataset = self.reduce_noise(processed_dataset, **self.pipeline["reduce_noise"])
        
                # other steps can be added here similarly

        return processed_dataset

    def resize_images(self, images, width, height):
        """Resizes the images to the specified dimensions.
        """
        pass

    def enhance_contrast(self, images):
        """Enhances the contrast of the images.

        """
        pass

        

    def reduce_noise(self, images, method='gaussian'):
        """Reduces noise in the images.
        """
        pass


    # Add other processing functions similarly...
