import tensorflow as tf

from source.image_preprocessing.preprocessing_steps.step_base import StepBase


class DummyStep(StepBase):
    """
    A dummy preprocessing step that does nothing to the image.
    """
    name = 'DummyStep'
    arguments_datatype = {}

    def __init__(self):
        """
        Initializes the DummyStep object for integration into an image preprocessing pipeline.
        """
        super().__init__(locals())
    
    @StepBase._tensor_pyfunc_wrapper
    def process_step(self, image_tensor):
        return image_tensor


if __name__ == '__main__':
    step = DummyStep()
    print(step.get_step_json_representation()) 