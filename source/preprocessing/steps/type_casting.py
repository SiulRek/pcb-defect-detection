import tensorflow as tf
from source.preprocessing.helpers.step_base import StepBase


class TypeCaster(StepBase):
    """A preprocessing step that casts an image tensor to a specified data type."""
    arguments_datatype = {'output_dtype': str}
    name = 'Type Caster'

    def __init__(self, output_dtype='float16'):
        """Initializes the TypeCaster object for integration into an image preprocessing pipeline.

           Args:
               output_dtype (str): The desired data type to cast the image tensor to.
                    Must be an attribute in tensorflow . Default is 'float16'.
        """
        super().__init__(locals())
        self.output_datatype = getattr(tf, output_dtype)

    @StepBase._tensor_pyfunc_wrapper
    def process_step(self, image_tensor):
        # image_tensor = tf.cast(image_tensor, self.output_datatype) Already done by the wrapper.
        return image_tensor


if __name__ == '__main__':
    step = TypeCaster()
    print(step.get_step_json_representation())
