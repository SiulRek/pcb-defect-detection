import tensorflow as tf

from source.preprocessing.helpers.step_base import StepBase


class MeanNormalizer(StepBase):
    """
    A preprocessing step that applies mean normalization to an image tensor.

    Note: The data type of the output image tensor is tf.float16.
    """
    arguments_datatype = {}
    name = 'Mean Normalizer'

    def __init__(self):
        """Initializes the MeanNormalizer object for integration into an image preprocessing
        pipeline."""
        super().__init__({})
        self.output_datatype = tf.float16

    @StepBase._tensor_pyfunc_wrapper
    def process_step(self, image_tensor):
        image_tensor = tf.cast(image_tensor, self.output_datatype)
        mean_val = tf.reduce_mean(image_tensor)
        range_val = tf.reduce_max(image_tensor) - tf.reduce_min(image_tensor)
        normalized_image = (image_tensor - mean_val) / (range_val + 1e-8)
        return normalized_image

if __name__ == '__main__':
    step = MeanNormalizer()
    print(step.get_step_json_representation())
