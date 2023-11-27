from tensorflow import float32 as tf_float32 
from tensorflow import reduce_min, reduce_max

from source.image_preprocessing.preprocessing_steps.step_base import StepBase

class MinMaxNormalizer(StepBase):
    """A preprocessing step that applies a min max normalization an image tensor."""

    arguments_datatype = {}
    name = 'Min Max Normalizer'

    def __init__(self):
        """Initializes the MinMaxNormalizer object for integration into an image preprocessing pipeline.
        """
        super().__init__({})
    
    def _set_output_datatypes(self):
        super()._set_output_datatypes()
        self._output_datatypes['image'] = tf_float32

    @StepBase._tf_function_decorator
    def process_step(self, image_tensor):
        min_val = reduce_min(image_tensor)
        max_val = reduce_max(image_tensor)
        normalized_image = (image_tensor - min_val) / (max_val - min_val)
        return normalized_image

if __name__ == '__main__':
    step = MinMaxNormalizer()
    print(step.get_step_json_representation())
    