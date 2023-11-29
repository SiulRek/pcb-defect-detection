from abc import ABC, abstractmethod
import tensorflow as tf

from source.image_preprocessing.preprocessing_steps.step_utils import correct_tf_image_shape

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
    - process_step(image: tf.Tensor or np.array) -> tf.Tensor or np.array:
        To be implemented by the child class to define the specific preprocessing functionality. The image datatype depends
        on the function_decorator used, either `tf_function_decorator`  for processing tensors  or `py_function_decorator`
        for processing np.arrays.


    Child Class Template:
        class StepTemplate(StepBase):

            arguments_datatype = <dictionary specifying the argument datatypes>
            name = <Preprocessing step identifier>
            def __init__(self, **processing_step_specific_args):
                super().__init__(locals())

            @StepBase._py_function_decorator # or @StepBase._tf_function_decorator depending on use case.
            def process_step(self, image_tensor):
                # TODO
                image_tensor_processed = ...
                return image_tensor_processed

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

    def get_step_json_representation(self):
        """Returns strings that corresponds to JSON entry text to be added to a JSON file."""

        # Convert datatype of values of params to match JSON format
        conv_params = {}
        for key, value in self._params.items():
            if isinstance(value, tuple):
                value = list(value)
            conv_params[key] = [value]

        params_str = ',\n'.join([f'        "{k}": {str(v).replace("True", "true").replace("False", "false")}' for k, v in conv_params.items()])
        json_string = f'    "{self.name}": {{\n{params_str}\n    }}' 
        return json_string
    
    @staticmethod
    def _tf_function_decorator(func):
        """ 
        A decorator for mapping TensorFlow tensor-based functions onto a dataset using tf.py_function.
        This decorator is designed for preprocessing steps implemented as Python functions,
        where the input to the function is a TensorFlow tensor.
        """
        def tensor_to_py_function_wrapper(self, image_tensor, target_tensor):
            processed_image = func(self, image_tensor)
            processed_image = tf.convert_to_tensor(processed_image, dtype=self._output_datatypes['image'])
            return processed_image, target_tensor

        def dataset_map_function(self, image_dataset):
            return image_dataset.map(
                lambda img, tgt: tf.py_function(
                    func=lambda i, t: tensor_to_py_function_wrapper(self, i, t),
                    inp=[img, tgt],
                    Tout=(self._output_datatypes['image'], self._output_datatypes['target'])
                )
            )

        return dataset_map_function

    @staticmethod
    def _py_function_decorator(func):
        """ 
        A decorator for mapping Python functions (processing NumPy arrays) onto a TensorFlow dataset using tf.py_function.
        This decorator is useful for preprocessing steps implemented in Python,
        converting TensorFlow tensors to NumPy arrays before processing.
        """
        def numpy_to_py_function_wrapper(self, image_tensor, target_tensor):
            image_nparray = image_tensor.numpy().astype('uint8')
            processed_image = func(self, image_nparray)
            processed_image = tf.convert_to_tensor(processed_image, dtype=self._output_datatypes['image'])
            processed_image = correct_tf_image_shape(processed_image)
            return processed_image, target_tensor

        def py_function_dataset_map(self, image_dataset):
            return image_dataset.map(
                lambda img, tgt: tf.py_function(
                    func=lambda i, t: numpy_to_py_function_wrapper(self, i, t),
                    inp=[img, tgt],
                    Tout=(self._output_datatypes['image'], self._output_datatypes['target'])
                )
            )

        return py_function_dataset_map

    @abstractmethod    
    def process_step(self, tf_image, tf_target):
        # Child class must implement this method.
        pass
