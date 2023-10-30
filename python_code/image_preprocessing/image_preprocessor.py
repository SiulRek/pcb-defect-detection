import tensorflow as tf
from copy import deepcopy

from python_code.image_preprocessing.image_preprocessing_steps import StepBase

class ImagePreprocessor:
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

        processed_dataset = image_dataset
        for step in self.pipeline:
            processed_dataset = step.process_step(processed_dataset)

        return processed_dataset
