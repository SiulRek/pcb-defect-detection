import tensorflow as tf

from source.image_preprocessing.preprocessing_steps.step_base import StepBase
from source.image_preprocessing.preprocessing_steps.step_utils import reduce_std


class StandardNormalizer(StepBase):
    """A preprocessing step that applies standard normalization (Z-score normalization) to an image tensor."""

    arguments_datatype = {}
    name = 'Standard Normalizer'

    def __init__(self):
        """Initializes the StandardNormalizer object for integration into an image preprocessing pipeline."""
        super().__init__({})
    
    def _set_output_datatypes(self):
        super()._set_output_datatypes()
        self._output_datatypes['image'] = tf.float32

    @StepBase._tf_function_decorator
    def process_step(self, image_tensor):
        image_tensor = tf.cast(image_tensor, self._output_datatypes['image'])
        mean_val = tf.reduce_mean(image_tensor)
        std_val = reduce_std(image_tensor)
        normalized_image = (image_tensor - mean_val) / (std_val + 1e-8)  # Added epsilon to avoid division by zero
        return normalized_image

if __name__ == '__main__':
    step = StandardNormalizer()
    print(step.get_step_json_representation())
