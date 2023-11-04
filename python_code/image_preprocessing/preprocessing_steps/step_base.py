import random
import json
import os

import tensorflow as tf

from python_code.utils.recursive_type_conversion import recursive_type_conversion

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..','..')
JSON_DEFAULT_PATH = os.path.join(ROOT_DIR, r'python_code/image_preprocessing/config/parameter_ranges.json')


class StepBase:
    """  Base class for defining preprocessing steps for images.

    Public Class Method:
    - set_json_path(filepath:str): Sets JSON path for configuration. Otherwise it defaults to `JSON_DEFAULT_PATH`.

    Public Instance Attributes (read-ony):
    - name (str): A name identifier for the preprocessing step (more than one consecutive underscores is not allowed!).
    - params (dict):  A dictionary containing parameters needed for the preprocessing step.

    Public Methods:
    - process_step(tf_image: tf.Tensor, tf_target: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        To be implemented by the child class to define the specific preprocessing functionality.

    
    Notes:
    - The class represents the base class for specific image preprocessing steps inheriting from this class.
    - Each child class must implement the `process_step` method.
    - The JSON file path for parameter randomization is defined as a constant JSON_PATH inside StepBase.

    Child Class Template:
        class StepTemplate(StepBase):

            def __init__(self, set_params_from_range=False, name_postfix='', **processing_step_specific_args):
                super().__init__('<Step name>', locals())

            @StepBase._py_function_decorator # or @StepBase._tf_function_decorator depending on use case.
            def process_step(self, tf_image, tf_target):
                # TODO
                tf_image_processed = ...
                return (tf_image_processed, tf_target)

    TODOs when integrating a new preprocessing step in the framework:
        1. Create preprocessing step class inheriting from `StepBase` according to template.
        2. Add mapping of the class to the constant `STEP_CLASS_MAPPING` {<self._name>:type(self)}.
        3. Add JSON entry of the class to parameter_ranges.json
    """
    
    _json_path = JSON_DEFAULT_PATH

    @classmethod
    def set_json_path(cls, path):
        """ Set the path to the JSON file specifieng the ranges of the preprocessing step parameters."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find specified JSON file with path '{path}'.")
        cls._json_path = path

    def __init__(self, name,  local_vars):
        """    Constructs the base preprocessing step with a customizable name and set of parameters.

        This method serves as the foundational setup for all inherited preprocessing step classes. 
        It integrates a unique identifier for each step and prepares the necessary parameters that 
        dictate the behavior of the specific image preprocessing routine. It is designed to be flexible, 
        allowing derived classes to pass in specific arguments that define the preprocessing step's unique 
        characteristics and operational parameters.

        Args:
            name (str): The base identifier for the preprocessing step.
            local_vars (dict): A collection of variables provided by the child class instantiation that includes configurations and a potential name postfix to append to the step's base name. Note: `local_vars` dict MUST contain the keys 'set_params_from_range' (value boolean) and 'name_postfix' (value string).
        """

        if 'name_postfix' not in local_vars.keys() or 'set_params_from_range' not in local_vars.keys():
            raise AttributeError("'name_postfix' or/and 'set_params_from_range' not in local vars, probably missing in child class parameter initialization.")
        
        self._name = name + local_vars['name_postfix']
        self._params = self._extract_params(local_vars)
        self._output_datatypes = {'image': None, 'target': None}
        self._set_output_datatypes()
        
    @property
    def name(self):
        """The name property is read-only."""
        return self._name

    @property
    def params(self):
        """The params property is read-only."""
        return self._params
    
    def __eq__(self, obj: 'StepBase') -> bool:
        return self._name.split('__')[0] == obj.name.split('__')[0] and self._params == obj.params

    def __str__(self):
        """The string representation of the class is at the same time the JSON entry text to be added to a JSON file."""

        # Convert datatype of values of params to match JSON format
        conv_params = {}
        for key, value in self._params.items():
            if isinstance(value, tuple):
                value = list(value)
            conv_params[key] = [value]

        params_str = ',\n'.join([f'        "{k}": {str(v).replace("True", "true").replace("False", "false")}' for k, v in conv_params.items()])
        json_string = f'    "{self._name}": {{\n{params_str}\n    }}'
        print(json_string)
        
    def _extract_params(self, local_vars):
        """  Extracts parameters needed for the preprocessing step based on local variables. It considers if parameters should be randomized or extracted directly from `local_vars`."""

        excluded_params = ['self', 'set_params_from_range', '__class__', 'name_postfix']
        initialization_params =  {key: value for key, value in local_vars.items() if key not in excluded_params}

        if local_vars['set_params_from_range']:
            return self._params_from_range(initialization_params)
        return initialization_params

    def _params_from_range(self, initialization_params): 
        """  Randomizes parameters for the preprocessing step based on value ranges defined in a JSON file. (Initialization parameters are required for datatype reference)."""

        configs = self._load_params_from_json()

        params = {}
        for key, value in initialization_params.items():

            if key not in configs:
                raise KeyError(f"JSON Configuration for instance named '{self.name}' does not contain the parameter '{key}'.")       

            params[key] = recursive_type_conversion(random.choice(configs[key]), value)  # Match the datatype of value.

        return params
    
    def _load_params_from_json(self):
        """   Loads parameters available for randomization from a JSON file. If a parameter for the current 
        preprocessing step is not available in the JSON, it raises a KeyError (The loaded value is converted to a datatype that matches the input_parameter)."""
        with open(StepBase._json_path, 'r', encoding='utf-8') as file:
            configs = json.load(file)
        return configs.get(self._name, {})
    
    def _set_output_datatypes(self):
        """ Sets the output datatypes of the step process."""
        # Child class can overwrite this method, otherwise defaults to the following:
        self._output_datatypes['image'] = tf.uint8
        self._output_datatypes['target'] = tf.int8
        
    def process_step(self, tf_image, tf_target):
        # Child class must implement this method.
        pass
    
    @staticmethod
    def _tf_function_decorator(func):       
        """ A decorator to wrap TensorFlow functions for mapping onto a dataset. Allows preprocessing step with tensorflow functionality a straigth forward implementation of the child classes."""
        def wrapper(self, image_dataset):
            def mapped_function(img, tgt):
                return func(self, img, tgt)
            return image_dataset.map(mapped_function)
        return wrapper

    @staticmethod
    def _py_function_decorator(func):
        """ A decorator to wrap python functions for mapping onto a dataset using tf.py_function. Allows preprocessing step with tensorflow functionality a straigth forward implementation of the child classes."""
        def wrapper(self, image_dataset):
            def mapped_function(img, tgt):
                processed_img, processed_tgt = tf.py_function(
                    func=lambda image, target: func(self, image, target),  # Lambda is used to pass self.
                    inp=[img, tgt],
                    Tout=(self._output_datatypes['image'], self._output_datatypes['target']),
                )
                return processed_img, processed_tgt
            return image_dataset.map(mapped_function)
        return wrapper

