import random
import json
import os

import cv2
import tensorflow as tf

from python_code.utils.recursive_type_conversion import recursive_type_conversion

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..','..')
JSON_DEFAULT_PATH = os.path.join(ROOT_DIR, r'python_code/image_preprocessing/config/parameter_ranges.json')

# TODOs when integrating a new preprocessing step in the framework:
    # 1. Create preprocessing step class inheriting from StepBase according to conventions.
    # 2. Add mapping of the class to the constant STEP_CLASS_MAPPING {<self.name>:type(self)}.
    # 3. Add json entry of the class to parameter_ranges.json

class StepBase:
    """  Base class for defining preprocessing steps for images.

    Class Attribute:
    - _json_path (str): Specifies the .json path to load configuration. Has Defaults to 'JSON_DEFAULT_PATH'.
    Instance Attributes:
    - name (str): A name identifier for the preprocessing step (more than one consecutive underscores is not allowed!).
    - params (dict):  A dictionary containing parameters needed for the preprocessing step.
    - output_datatypes (dict): A dictionary containing the output datatypes (Only relevand when using the py_function_decorator).

    Methods:
    - process_step(tf_image: tf.Tensor, tf_target: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        To be implemented by the child class to define the specific preprocessing functionality.

    - correct_shape() -> tf.Tensor:
        Corrects the shape of a TensorFlow image tensor based on the inferred dimensions.

    - print_json_entry():
        Prints the json entry corresponding to the attributes 'name' and 'params' of the created instance (To be added manually in the json file).
    
    - _set_output_datatypes():
        Function to set the output_datatypes(), child classes are allowed to overwrite the function.
          
    - _extract_params(local_vars: dict) -> dict:
        Extracts parameters needed for the preprocessing step based on local variables. 
        It considers if parameters should be randomized or extracted directly from local_vars.

    - _tf_function_decorator(func: Callable) -> Callable:
        A decorator to wrap TensorFlow functions for mapping onto a dataset.

    - _py_function_decorator(func: Callable) -> Callable:
        A decorator to wrap python functions for mapping onto a dataset using tf.py_function.

    - _params_from_range() -> None:
        Randomizes parameters for the preprocessing step based on value ranges defined in a JSON file.

    - _load_params_from_json() -> dict:
        Loads parameters available for randomization from a JSON file. If a parameter for the current 
        preprocessing step is not available in the JSON, it raises a KeyError (The loaded value is converted to a datatype that matches the input_parameter).


    Notes:
    - The class is represents the base class for specific preprocessing steps inheriting from this class.
    - Each child class must implement the `process_step` method and must execute super().init(<specific child class params>) in the __init__() method.
    - The JSON file path for parameter randomization is defined as a constant JSON_PATH.
    """
    
    _json_path = JSON_DEFAULT_PATH

    @classmethod
    def set_json_path(cls, path):
        """ Set the path to the json file specifieng the ranges of the preprocessing step parameters."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find specified json file with path '{path}'.")
        cls._json_path = path

    def __init__(self, name,  local_vars):
        self.name = name
        self.params = self._extract_params(local_vars)
        self.output_datatypes = {'image': None, 'target': None}
        self._set_output_datatypes()
        
    def __eq__(self, obj: 'StepBase') -> bool:
        return self.name == obj.name and self.params == obj.params

    def __str__(self):
        # The string representation of the class is at the same time the json entry text to be added to a .json file.

        # Convert datatype of values of params to match json format
        conv_params = {}
        for key, value in self.params.items():
            if isinstance(value, tuple):
                value = list(value)
            conv_params[key] = [value]

        params_str = ',\n'.join([f'        "{k}": {str(v).replace("True", "true").replace("False", "false")}' for k, v in conv_params.items()])
        json_string = f'    "{self.name}": {{\n{params_str}\n    }}'
        print(json_string)
        
    def _extract_params(self, local_vars):
        """ Extracts the configuration parameters from the initialization parameters or from .json file if 'set_params_from_range' is true."""

        initialization_params =  {key: value for key, value in local_vars.items() if key not in ['self', 'set_params_from_range', '__class__']}

        if local_vars['set_params_from_range']:
            return self._params_from_range(initialization_params)
        return initialization_params

    def _params_from_range(self, initialization_params): 
        """ Returns parameter from the specified .json file. (Initialization parameters are required for datatype reference)."""

        configs = self._load_params_from_json()

        params = {}
        for key, value in initialization_params.items():

            if key not in configs:
                raise KeyError(f"JSON Configuration for class '{str(self)}' does not contain the parameter '{key}'.")       

            params[key] = recursive_type_conversion(random.choice(configs[key]), value)  # Match the datatype of value.

        return params
    
    def _load_params_from_json(self):
        with open(StepBase._json_path, 'r', encoding='utf-8') as file:
            configs = json.load(file)
        return configs.get(self.name, {})
    
    def _set_output_datatypes(self):
        # Child class can overwrite this method^, otherwise defaults to the following:
        self.output_datatypes['image'] = tf.uint8
        self.output_datatypes['target'] = tf.int8
        
    def process_step(self, tf_image, tf_target):
        # Child class must implement this method.
        pass
    
    @staticmethod
    def _tf_function_decorator(func):       
        # Allows preprocessing step with tensorflow functionality a straigth forward implementation of the child classes.
        def wrapper(self, image_dataset):
            def mapped_function(img, tgt):
                return func(self, img, tgt)
            return image_dataset.map(mapped_function)
        return wrapper

    @staticmethod
    def _py_function_decorator(func):
        # Allows preprocessing step with python functionality a straigth forward implementation of the  child classes .
        def wrapper(self, image_dataset):
            def mapped_function(img, tgt):
                processed_img, processed_tgt = tf.py_function(
                    func=lambda image, target: func(self, image, target),  # Lambda is used to pass self.
                    inp=[img, tgt],
                    Tout=(self.output_datatypes['image'], self.output_datatypes['target']),
                )
                return processed_img, processed_tgt
            return image_dataset.map(mapped_function)
        return wrapper


    def correct_shape(self, tf_image):
        """
        Corrects the shape of a TensorFlow image tensor based on the inferred dimensions.
        
        Parameters:
        - tf_image (tf.Tensor): The input image tensor.
        
        Returns:
        - tf.Tensor: A reshaped tensor based on inferred dimensions.
        """
        
        height = tf.shape(tf_image)[0]
        width = tf.shape(tf_image)[1]
        channel_num = tf.shape(tf_image)[2]
        
        reshaped_image = tf.reshape(tf_image, [height, width, channel_num])
        
        return reshaped_image    
    