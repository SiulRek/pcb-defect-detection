import json
import re

from copy import deepcopy
import tensorflow as tf

from python_code.image_preprocessing.preprocessing_steps.step_base import StepBase
from python_code.image_preprocessing.preprocessing_steps.step_class_mapping import STEP_CLASS_MAPPING

class ImagePreprocessor:
    """
    Manages and processes a pipeline of image preprocessing steps for PCB images.

    The ImagePreprocessor class encapsulates a sequence of preprocessing operations
    defined as steps. Each step is a discrete preprocessing action, such as noise
    reduction, normalization, etc. The steps are applied in sequence to an input
    dataset of images.

    Attributes:
        pipeline (list of StepBase): A list of preprocessing steps to be executed.

    Methods:
        pipeline(self):
            Property that returns the current pipeline.

        set_pipe(self, pipeline: List[StepBase]):
            Sets the preprocessing pipeline with a deep copy of the provided steps,
            ensuring each step is an instance of a StepBase subclass.

        add_step(self, step: StepBase):
            Adds a new step to the pipeline, verifying that it is a subclass of StepBase.

        process(self, image_dataset: tf.data.Dataset) -> tf.data.Dataset:
            Applies each preprocessing step to the provided dataset and returns the processed dataset.

        save_pipe_to_json(self, filepath: str):
            Serializes the preprocessing pipeline to a JSON file, saving the step configurations.

        load_pipe_from_json(self, filepath: str):
            Loads and reconstructs a preprocessing pipeline from a JSON file.

    Notes:
        - The pipeline should only contain instances of classes that inherit from StepBase.
        - The `set_pipe` and `add_step` methods include type checks to enforce this.
        - The JSON serialization and deserialization methods (`save_pipe_to_json` and `load_pipe_from_json`)
          handle the conversion and reconstruction of the pipeline steps, respectively.
        - The `process` method's behavior changes based on the `raise_step_process_exception` flag,
                allowing for flexible error handling during the preprocessing stage.
    
    """

    def __init__(self, raise_step_process_exception=True): 
        """ Initializes the ImagePreprocessor with an empty pipeline.
            The `raise_step_process_exception` flag determines whether exceptions
            during step processing are raised or logged.
        """
        self._pipeline = []
        self._raise_step_process_exception = raise_step_process_exception

    @property
    def pipeline(self):
        return self._pipeline

    def set_pipe(self, pipeline):
        """  Sets the preprocessing pipeline with a deep copy of the provided steps ensuring each step is an instance of a StepBase subclass."""
        for step in pipeline:
            if not isinstance(step, StepBase):  
                raise ValueError(f'Expecting a Child of StepBase, got {type(step)} instead.')        
        self._pipeline = deepcopy(pipeline)

    def add_step(self, step):
        """ Adds a new step to the pipeline, verifying that it is a subclass of StepBase."""
        if not isinstance(step, StepBase):  
                    raise ValueError(f'Expecting a Child of StepBase, got {type(step)} instead.')
        self._pipeline.append(deepcopy(step))

    def process(self, image_dataset):
        """  Applies each preprocessing step to the provided dataset and returns the processed dataset.
            If `_raise_step_process_exception` is True, exceptions in processing a step will be caught and logged,
            and the process will return None. If False, it will proceed without exception handling.
        """
        processed_dataset = image_dataset
        for step in self.pipeline:
            if not self._raise_step_process_exception:
                processed_dataset = step.process_step(processed_dataset)
            else:
                try:
                    processed_dataset = step.process_step(processed_dataset)
                except Exception as e:
                    print(f"An error occurred in step {step.name}: {str(e)}")
                    return None

        return processed_dataset

    def save_pipe_to_json(self, filepath):
        "Serializes the preprocessing pipeline to a JSON file, saving the step configurations."

        json_data = {}
        for step in self.pipeline:

            converted_params = {}
            for key, value in step.params.items():
                converted_params[key] = [self._convert_tuple_to_list(value)]  # Square Brackets required as StepBase child instances expect ranges.
            
            name = self._generate_unique_entry_name(step.name, json_data)   
            json_data[name] = converted_params
        
        json_string = json.dumps(json_data, indent=4)
        json_string = json_string.replace('},', '},\n')

        pattern = r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]'   # This regex pattern finds text within square brackets, including nested brackets
        result = re.sub(pattern, self._remove_newlines, json_string)  # Replace newlines and spaces within square brackets (improves readability)

        with open(filepath, 'w') as file:
            file.write(result)
    
    def _convert_tuple_to_list(self, obj, recursive_call=False):
        """ Helper method to recursively convert tuples in a nested structure to lists."""
        if isinstance(obj, tuple) or isinstance(obj, list):
            return [self._convert_tuple_to_list(item, recursive_call=True) for item in obj]
        if type(obj) in {int, float, str, bool}:
            return obj
        else:
            raise TypeError(f"Object with value '{obj} cannot not be recursivly converted to list.")
    
    def _generate_unique_entry_name(self, current_entry, json_data):        
        """ Generates a unique step name for JSON entries to avoid conflicts."""
        name = current_entry
        i = 2
        while name in json_data.keys():              # Same namining of entries are not allowed in json.
            name = name.split('__')[0] + '__' + str(i)
            i += 1

        return name
        
    def _remove_newlines(self, match):
        """ Removes newlines and spaces within square brackets in JSON strings."""
        return match.group().replace('\n', '').replace(' ', '')
    
        
    def load_pipe_from_json(self, filepath):
        """  Loads and reconstructs a preprocessing pipeline from a JSON file.
        """
        
        StepBase.set_json_path(filepath)
        with open(filepath, 'r') as file:
            data = json.load(file)
            step_names = list(data.keys())
        
        self.pipeline.clear()
        for step_name in step_names:
            step = self._construct_step_instance(step_name)
            self.add_step(step)
    
    def _construct_step_instance(self, step_name):
        """Constructs an instance of a preprocessing step from its name, as defined in the JSON configuration. """    

        step_name_parts = step_name.split('__')

        if step_name_parts[0] not in STEP_CLASS_MAPPING.keys():
            raise KeyError(f"Step name {step_name_parts[0]} from json file has no mapping.")
        step_class = STEP_CLASS_MAPPING[step_name_parts[0]]

        if len(step_name_parts) > 1:
            name_postfix = '__' + '__'.join(step_name_parts[1:])
            return step_class(set_params_from_range=True, name_postfix=name_postfix)
        else:
            return step_class(set_params_from_range=True)

