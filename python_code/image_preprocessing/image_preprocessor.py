import json
import re

from copy import deepcopy
import tensorflow as tf

from python_code.image_preprocessing.preprocessing_steps.step_base import StepBase
from python_code.image_preprocessing.preprocessing_steps.step_class_mapping import STEP_CLASS_MAPPING

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

    def save_pipe_to_json(self, filepath):

        json_data = {}
        for step in self.pipeline:

            converted_params = {}
            for key, value in step.params.items():
                converted_params[key] = [self._convert_tuple_to_list(value)]  # Required as StepBase child instances expect ranges.
            
            name = step.name           
            i = 2
            while name in json_data.keys():              # Same namining of entries are not allowed in json.
                name = name.split('__')[0] + '__' + str(i)
                i += 1
                
            json_data[name] = converted_params
        
        json_string = json.dumps(json_data, indent=4)
        json_string = json_string.replace('},', '},\n')

        pattern = r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]'   # This regex pattern finds text within square brackets, including nested brackets
        result = re.sub(pattern, self._remove_newlines, json_string)  # Replace newlines and spaces within square brackets (improves readability)

        with open(filepath, 'w') as file:
            file.write(result)
    
    def _remove_newlines(self, match):
        return match.group().replace('\n', '').replace(' ', '')
    
    def _convert_tuple_to_list(self, obj, recursive_call=False):
        """Recursively converts all tuples within a nested structure of lists, tuples, to lists."""
        if isinstance(obj, tuple) or isinstance(obj, list):
            return [self._convert_tuple_to_list(item, recursive_call=True) for item in obj]
        if type(obj) in {int, float, str, bool}:
            return obj
        else:
            raise TypeError(f"Object with value '{obj} cannot not be recursivly converted to list.")
        
    def load_pipe_from_json(self, filepath):
        
        StepBase.set_json_path(filepath)
        with open(filepath, 'r') as file:
            data = json.load(file)
            step_names = list(data.keys())
        
        self.pipeline.clear()
        for step_name in step_names:
            step_name = step_name.split('__')[0]
            if step_name not in STEP_CLASS_MAPPING.keys():
                raise KeyError(f"Step name {step_name} from json file has no mapping.")
            step_class = STEP_CLASS_MAPPING[step_name]
            step = step_class(set_params_from_range=True)
            self.add_step(step)

