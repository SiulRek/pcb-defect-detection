import tensorflow as tf
from source.image_preprocessing.preprocessing_steps.step_base import StepBase

class SquareShapePadder(StepBase):
    """ A preprocessing step that pads an image to a square shape using a specified pixel value. """

    arguments_datatype = {'padding_pixel_value': int}
    name = 'Square Shape Padder'

    def __init__(self, padding_pixel_value=0):
        """ Initializes the SquareShapePadder object for integration into an image preprocessing pipeline.

        Args:
            padding_pixel_value (int, optional): The pixel value to be used for padding. Defaults to 0.
        """
        super().__init__(locals())

    @StepBase._tf_function_decorator
    def process_step(self, image_tensor):
        shape = tf.shape(image_tensor)
        height, width = shape[0], shape[1]

        if height > width:
            pad_top = 0
            pad_bottom = 0
            pad_left = (height - width) // 2
            pad_right = height - width - pad_left
        else:
            pad_top = (width - height) // 2
            pad_bottom = width - height - pad_top
            pad_left = 0
            pad_right = 0

        tf_padded_img = tf.pad(image_tensor, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=self.params['padding_pixel_value'])

        return tf_padded_img

if __name__ == '__main__':
    step = SquareShapePadder()
    print(step.get_step_json_representation())