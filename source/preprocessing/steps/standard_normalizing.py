import tensorflow as tf

from source.preprocessing.helpers.for_steps.step_base import StepBase
from source.preprocessing.helpers.for_steps.step_utils import reduce_std


class StandardNormalizer(StepBase):
    """
    A preprocessing step that applies standard normalization (Z-score normalization) to an image
    tensor.

    Note: The data type of the output image tensor is tf.float16.
    """
    arguments_datatype = {}
    name = 'Standard Normalizer'

    def __init__(self):
        """Initializes the StandardNormalizer object for integration into an image preprocessing
        pipeline."""
        super().__init__({})
        self.output_datatype = tf.float16

    @StepBase._tensor_pyfunc_wrapper
    def process_step(self, image_tensor):
        image_tensor = tf.cast(image_tensor, self.output_datatype)
        mean_val = tf.reduce_mean(image_tensor)
        std_val = reduce_std(image_tensor)
        normalized_image = (image_tensor - mean_val) / (std_val + 1e-8)
        return normalized_image


if __name__ == '__main__':
    step = StandardNormalizer()
    print(step.get_step_json_representation())
