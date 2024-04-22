import tensorflow as tf

from source.preprocessing.preprocessing_steps.step_base import StepBase


class Rotator(StepBase):
    """
    A preprocessing step that rotates an image tensor by a specified angle.
    The angle of rotation is specified as an input parameter.

    Note: the angle of rotation must be a multiple of 90 degrees. Otherwise, the angle will be
        rounded to the nearest multiple of 90 degrees.
    """

    arguments_datatype = {'angle': float}
    name = 'Rotator'

    def __init__(self, angle=90.0):
        """
        Initializes the Rotator object for integration into an image preprocessing pipeline.

        Args:
            angle (float): The angle of rotation in degrees. Default is 90.0.
        """
        super().__init__(locals())

    @StepBase._tensor_pyfunc_wrapper
    def process_step(self, image_tensor):
        rotated_image = tf.image.rot90(image_tensor, k=int(self.parameters['angle'] / 90))
        return rotated_image


if __name__ == '__main__':
    step = Rotator()
    print(step.get_step_json_representation())