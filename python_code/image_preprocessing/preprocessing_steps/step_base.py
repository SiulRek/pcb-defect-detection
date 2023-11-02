import random
import json
import os

import cv2
import tensorflow as tf

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..','..')
JSON_PATH = os.path.join(ROOT_DIR, r'python_code/image_preprocessing/config/parameter_ranges.json')

class StepBase:
    """  Base class for defining preprocessing steps for images.

    Attributes:
    - name (str): A name identifier for the preprocessing step.
    - params (dict):  A dictionary containing parameters needed for the preprocessing step.
    - output_datatypes (dict): A dictionary containing the output datatypes (Only relevand when using the py_function_decorator).

    Methods:
    - process_step(tf_image: tf.Tensor, tf_target: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        To be implemented by the child class to define the specific preprocessing functionality.
    - set_output_datatypes():
        Function to set the output_datatypes(), child classes are allowed to overwrite the function.
          
    - extract_params(local_vars: dict) -> dict:
        Extracts parameters needed for the preprocessing step based on local variables. 
        It considers if parameters should be randomized or extracted directly from local_vars.

    - tf_function_decorator(func: Callable) -> Callable:
        A decorator to wrap TensorFlow functions for mapping onto a dataset.

    - py_function_decorator(func: Callable) -> Callable:
        A decorator to wrap python functions for mapping onto a dataset using tf.py_function.

    - params_from_range() -> None:
        Randomizes parameters for the preprocessing step based on value ranges defined in a JSON file.

    - load_params_from_json() -> dict:
        Loads parameters available for randomization from a JSON file. If a parameter for the current 
        preprocessing step is not available in the JSON, it raises a KeyError.

    - correct_shape() -> tf.Tensor:
        Corrects the shape of a TensorFlow image tensor based on the inferred dimensions.

    Notes:
    - The class is represents the base class for specific preprocessing steps inheriting from this class.
    - Each child class must implement the `process_step` method and must execute super().init(<specific child class params>) in the __init__() method.
    - The JSON file path for parameter randomization is defined as a constant JSON_PATH.
    """

    def __init__(self, name,  local_vars):
        self.name = name
        self.params = self.extract_params(local_vars)
        self.output_datatypes = {'image': None, 'target': None}
        self.set_output_datatypes()

    def extract_params(self, local_vars):
        if local_vars['set_params_from_range']:
            return self.params_from_range()
        return {key: value for key, value in local_vars.items() if key not in ['self', 'set_params_from_range', '__class__']}

    def params_from_range(self): 

        configs = self.load_params_from_json()

        params = {}
        for key, value in self.params.items():
            if key not in configs:
                raise KeyError(f"Config for class '{str(self)}' does not contain the parameter '{key}'.")       
            type_of_value = type(value)
            params[key] = type_of_value(random.choice(configs[key]))

        return params
    
    def load_params_from_json(self):
        with open(JSON_PATH, 'r', encoding='utf-8') as file:
            configs = json.load(file)
        return configs.get(self.name, {})
    
    def set_output_datatypes(self):
        # Child class can overwrite this method
        self.output_datatypes['image'] = tf.uint8
        self.output_datatypes['target'] = tf.int8
        
    def process_step(self, tf_image, tf_target):
        # Child class must implement this method.
        pass
    
    @staticmethod
    def tf_function_decorator(func):
        def wrapper(self, image_dataset):
            def mapped_function(img, tgt):
                return func(self, img, tgt)
            return image_dataset.map(mapped_function)
        return wrapper

    @staticmethod
    def py_function_decorator(func):
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

