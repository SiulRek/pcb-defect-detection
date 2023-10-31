import tensorflow as tf
from copy import deepcopy

from python_code.image_preprocessing.image_preprocessing_steps import StepBase

class ImagePreprocessor:
    """ A class to define and process the PCB image preprocessing pipeline.

    Attributes:
    - pipeline (list)
        List containing the steps (which are children of StepBase) to be executed in the preprocessing pipeline.

    Methods:
    - add_step(step: StepBase) -> None:
        Adds a preprocessing step to the current pipeline.

    - process(image_dataset: tf.data.Dataset) -> tf.data.Dataset:
        Processes the provided image dataset through the defined preprocessing pipeline.

    Notes:
    - The pipeline is executed in the order the steps are added.
    - Each step in the pipeline should be an instance of a child class of StepBase.

    """
    def __init__(self, pipeline=None): 
        if pipeline is None:
            pipeline = []
        self._pipeline = deepcopy(pipeline)

    @property
    def pipeline(self):
        return self._pipeline

    @pipeline.setter
    def pipeline(self, pipeline):
        for step in pipeline:
            if not isinstance(step, StepBase):  
                raise ValueError(f'Expecting a Child of StepBase, got {type(step)} instead.')        
        self._pipeline = deepcopy(pipeline)

    def add_step(self, step):
        if not isinstance(step, StepBase):  
                    raise ValueError(f'Expecting a Child of StepBase, got {type(step)} instead.')
        self._pipeline.append(deepcopy(step))

    def process(self, image_dataset):

        #TODO: Add Error handling here.
        processed_dataset = image_dataset
        for step in self.pipeline:
            processed_dataset = step.process_step(processed_dataset)

        return processed_dataset
