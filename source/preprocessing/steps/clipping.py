import tensorflow as tf

from source.preprocessing.helpers.for_steps.step_base import StepBase

class Clipper(StepBase):
    """
    A preprocessing step that clips the pixel values of an image tensor to a specified range.
    """
    arguments_datatype = {'min_value': float, 'max_value': float}
    name = 'Clipper'

    def __init__(self, min_value=0.0, max_value=1.0):
        """
        Initializes the Clipper object for integration into an image preprocessing pipeline.

        Args:
            min_value (float): The minimum value to clip to. Default is 0.0.
            max_value (float): The maximum value to clip to. Default is 1.0.
        """
        super().__init__(locals())

    @StepBase._tensor_pyfunc_wrapper
    def process_step(self, image_tensor):
        image_tensor = tf.cast(image_tensor, self.output_datatype)
        min_value = tf.cast(self.parameters['min_value'], dtype=self.output_datatype)
        max_value = tf.cast(self.parameters['max_value'], dtype=self.output_datatype)
        clipped_image = tf.clip_by_value(image_tensor, min_value, max_value)
        return clipped_image

if __name__ == '__main__':
    step = Clipper()
    print(step.get_step_json_representation())
