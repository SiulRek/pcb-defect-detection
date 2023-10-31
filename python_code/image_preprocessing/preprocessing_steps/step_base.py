import random
import json
import os

import cv2
import tensorflow as tf

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
JSON_PATH = os.path.join(ROOT_DIR, r'python_code/image_preprocessing/parameter_ranges.json')

class StepBase:
    """  Base class for defining preprocessing steps for images.

    Attributes:
    - name (str): A name identifier for the preprocessing step.
    - params (dict):  A dictionary containing parameters needed for the preprocessing step.

    Methods:
    - process_step(tf_image: tf.Tensor, tf_target: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        To be implemented by the child class to define the specific preprocessing functionality.

    - extract_params(local_vars: dict) -> dict:
        Extracts parameters needed for the preprocessing step based on local variables. 
        It considers if parameters should be randomized or extracted directly from local_vars.

    - tf_function_decorator(func: Callable) -> Callable:
        A decorator to wrap TensorFlow functions for mapping onto a dataset.

    - py_function_decorator(func: Callable) -> Callable:
        A decorator to wrap python functions for mapping onto a dataset using tf.py_function.

    - random_params() -> None:
        Randomizes parameters for the preprocessing step based on values defined in a JSON file.

    - load_params_from_json() -> dict:
        Loads parameters available for randomization from a JSON file. If a parameter for the current 
        preprocessing step is not available in the JSON, it raises a KeyError.

    - reshape_color_channel(tf_image: tf.Tensor, color_channel: str = 'gray', tf_image_comparison: tf.Tensor = None) -> tf.Tensor:
        Helper method to reshape image tensor based on color channel. It supports reshaping to grayscale 
        or RGB. If tf_image_comparison is provided, it reshapes the image to match the channels of tf_image_comparison.

    Notes:
    - The class is represents the base class for specific preprocessing steps inheriting from this class.
    - Each child class must implement the `process_step` method and must execute super().init(<specific child class params>) in the __init__() method.
    - The JSON file path for parameter randomization is defined as a constant JSON_PATH.
    """

    def __init__(self, name,  local_vars):
        self.name = name
        self.params = self.extract_params(local_vars)
        
    def process_step(self, tf_image, tf_target):
        # Child class must implement this method.
        pass
    
    def extract_params(self, local_vars):
        if local_vars['set_random_params']:
            return self.random_params()
        return {key: value for key, value in local_vars.items() if key != 'self' and key != 'set_random_params'}

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
                    Tout=(tf.uint8, tf.int8),
                )
                return processed_img, processed_tgt
            return image_dataset.map(mapped_function)
        return wrapper

    def random_params(self): 

        configs = self.load_params_from_json()

        for key, value in self.params.items():
            if key not in configs:
                raise KeyError(f"Config for class '{str(self)}' does not contain the parameter '{key}'.")       
            type_of_value = type(value)
            self.params[key] = type_of_value(random.choice(configs[key]))

    def load_params_from_json(self):
        with open(JSON_PATH, 'r', encoding='utf-8') as file:
            configs = json.load(file)
        return configs.get(self.name, {})

    def reshape_color_channel(self, tf_image, color_channel='gray', tf_image_comparison=None):
        if tf_image_comparison is not None:
            return tf.reshape(tf_image, [tf_image.shape[0], tf_image.shape[1], tf_image_comparison.shape[2]])
        elif color_channel == 'gray':
            return tf.reshape(tf_image, [tf_image.shape[0], tf_image.shape[1], 1])
        elif color_channel == 'rgb':
            return tf.reshape(tf_image, [tf_image.shape[0], tf_image.shape[1], 3])
        else:
            raise ValueError(f'Color channel {color_channel} is invalid.')