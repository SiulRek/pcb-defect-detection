import tensorflow as tf

from source.image_preprocessing.preprocessing_steps.step_base import StepBase


class MinMaxNormalizer(StepBase):
    """
    A preprocessing step that applies a min max normalization an image tensor.

    Note: The data type of the output image tensor is tf.float16.
    """
    arguments_datatype = {}
    name = 'Min Max Normalizer'

    def __init__(self):
        """Initializes the MinMaxNormalizer object for integration into an image preprocessing
        pipeline.
        """
        super().__init__({})
        self.output_datatype = tf.float16

    @StepBase._tensor_pyfunc_wrapper
    def process_step(self, image_tensor):
        image_tensor = tf.cast(image_tensor, self.output_datatype)
        min_val = tf.reduce_min(image_tensor)
        max_val = tf.reduce_max(image_tensor)
        normalized_image = (image_tensor - min_val) / (max_val - min_val)
        return normalized_image

if __name__ == '__main__':
    step = MinMaxNormalizer()
    print(step.get_step_json_representation())
