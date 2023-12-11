import tensorflow as tf

from source.image_preprocessing.preprocessing_steps.step_base import StepBase


class LocalContrastNormalizer(StepBase):
    """
    A preprocessing step that applies local contrast normalization to an image tensor.    
    
    Note: The data type of the output image tensor is tf.float16.
    """

    arguments_datatype = {
        'depth_radius': int,
        'bias': float,
        'alpha': float,
        'beta': float
    }
    name = 'Local Contrast Normalizer'

    def __init__(self, depth_radius=5, bias=1.0, alpha=1e-4, beta=0.75):
        """Initializes the LocalContrastNormalizer object for integration into an image preprocessing pipeline.

        Args:
            depth_radius (int): Depth radius for normalization.
            bias (float): Bias to avoid division by zero.
            alpha (float): Scale factor.
            beta (float): Exponent for normalization.

        Note: 
            - This step is ideally applied to images that have already undergone standard normalization. This ensures that the image data is centered and scaled appropriately before local contrast enhancement.
        """
        super().__init__(locals())
        self.output_datatypes['image'] = tf.float16
    
    @StepBase._tensor_pyfunc_wrapper
    def process_step(self, image_tensor):
       
        image_tensor = tf.cast(image_tensor, tf.float16)

        # Add a batch dimension to image_tensor if it doesn't have one
        if len(image_tensor.shape) == 3:
            image_tensor = tf.expand_dims(image_tensor, axis=0)

        image_lcn = tf.nn.local_response_normalization(
            image_tensor,
            depth_radius=self.parameters['depth_radius'],
            bias=self.parameters['bias'],
            alpha=self.parameters['alpha'],
            beta=self.parameters['beta']
        )

        # Remove the batch dimension if it was added earlier
        if len(image_lcn.shape) == 4 and image_lcn.shape[0] == 1:
            image_lcn = tf.squeeze(image_lcn, axis=0)

        return image_lcn


if __name__ == '__main__':
    step = LocalContrastNormalizer()
    print(step.get_step_json_representation())
