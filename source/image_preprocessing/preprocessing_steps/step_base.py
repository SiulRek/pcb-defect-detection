from abc import ABC, abstractmethod
import os
import tensorflow as tf

from source.utils.recursive_type_conversion import recursive_type_conversion


class StepBase(ABC):
    """  Base class for defining preprocessing steps for images in a image preprocessing pipeline.

    This abstract class provides a structured approach to implementing various image preprocessing steps. 
    Each step is characterized by its unique parameters and functionality, which are defined in the child classes 
    inheriting from StepBase. The class facilitates the integration and execution of preprocessing steps within a 
    TensorFlow image processing pipeline.

    Child classes should implement the `process_step` method according to their specific processing requirements and
    specify the value of the class attributes: `arguments_datatype` and `name`.
    Decorators `_tf_function_decorator` and `_py_function_decorator` are provided for flexibility in implementing 
    TensorFlow and Python functions, respectively.

    Public Class Attribute (read-ony):
    - arguments_datatype (dict):  A dictionary containing the specification of the argument datatypes.
    - name (str): The base identifier for the preprocessing step.

    Public Instance Attribute (read-ony):
    - params (dict):  A dictionary containing parameters needed for the preprocessing step.

    Public Methods:
    - process_step(tf_image: tf.Tensor, tf_target: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        To be implemented by the child class to define the specific preprocessing functionality.


    Child Class Template:
        class StepTemplate(StepBase):

            arguments_datatype = <dictionary specifying the argument datatypes>
            name = <Preprocessing step identifier>
            def __init__(self, **processing_step_specific_args):
                super().__init__(locals())

            @StepBase._py_function_decorator # or @StepBase._tf_function_decorator depending on use case.
            def process_step(self, tf_image, tf_target):
                # TODO
                tf_image_processed = ...
                return (tf_image_processed, tf_target)

    TODOs when integrating a new preprocessing step in the framework:
        1. Create preprocessing step class inheriting from `StepBase` according to template.
        2. Add mapping of the class to the constant `STEP_CLASS_MAPPING` {<self._name>:type(self)}.
        3. Add JSON entry of the class to .source/image_preprocessing/pipeline/template.json
        4. Execute single_step_test.py over this class.
    """
    
    arguments_datatype = None   
    name = None

    def __init__(self, local_vars):
        """    Constructs the base preprocessing step with a customizable name and set of parameters.

        This method serves as the foundational setup for all inherited preprocessing step classes. 
        It integrates a unique identifier for each step and prepares the necessary parameters that 
        dictate the behavior of the specific image preprocessing routine. It is designed to be flexible, 
        allowing derived classes to pass in specific arguments that define the preprocessing step's unique 
        characteristics and operational parameters.

        Args:
            local_vars (dict): A collection of variables provided by the child class instantiation that includes hyperparameter configurations and.
        """
        self._params = self._extract_params(local_vars)
        self._output_datatypes = {'image': None, 'target': None}
        self._set_output_datatypes()
        
    @property
    def params(self):
        """The params property is read-only."""
        return self._params
    
    def __eq__(self, obj: 'StepBase') -> bool:
        return self.name.split('__')[0] == obj.name.split('__')[0] and self._params == obj.params

    def __str__(self):
        """The string representation of the class is at the same time the JSON entry text to be added to a JSON file."""

        # Convert datatype of values of params to match JSON format
        conv_params = {}
        for key, value in self._params.items():
            if isinstance(value, tuple):
                value = list(value)
            conv_params[key] = [value]

        params_str = ',\n'.join([f'        "{k}": {str(v).replace("True", "true").replace("False", "false")}' for k, v in conv_params.items()])
        json_string = f'    "{self.name}": {{\n{params_str}\n    }}' 
        return json_string
        
    def _extract_params(self, local_vars):
        """  Extracts parameters needed for the preprocessing step based on local variables. It considers if parameters should be randomized or extracted directly from `local_vars`."""

        excluded_params = ['self', '__class__']
        initialization_params =  {key: value for key, value in local_vars.items() if key not in excluded_params}

        return initialization_params
    
    def _set_output_datatypes(self):
        """ Sets the output datatypes of the step process."""
        # Child class can overwrite this method, otherwise defaults to the following:
        self._output_datatypes['image'] = tf.uint8
        self._output_datatypes['target'] = tf.int8

    @abstractmethod    
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

