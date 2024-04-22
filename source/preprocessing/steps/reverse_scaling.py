import tensorflow as tf
from source.preprocessing.helpers.for_steps.step_base import StepBase

class ReverseScaler(StepBase):
    """
    A preprocessing step that scales an image tensor by a specified factor.
       The default scale factor is 255, commonly used to normalize 0-255 RGB images to a 0-1 range.


    Note: The data type of the output image tensor is tf.float16.
    """

    arguments_datatype = {'scale_factor': float}
    name = 'Reverse Scaler'

    def __init__(self, scale_factor=255):
        """
        Initializes the ReverseScaler object for integration into an image preprocessing
        pipeline.

           Args:
               scale_factor (float): The factor used for scaling the image tensor. Default is 255.
        """
        super().__init__(locals())
        self.output_datatype = tf.float16

    @StepBase._tensor_pyfunc_wrapper
    def process_step(self, image_tensor):
        image_tensor = tf.cast(image_tensor, self.output_datatype)
        scale_factor = tf.constant(self.parameters['scale_factor'], self.output_datatype)
        scaled_image = image_tensor / scale_factor
        return scaled_image


if __name__ == '__main__':
    step = ReverseScaler()
    print(step.get_step_json_representation())
