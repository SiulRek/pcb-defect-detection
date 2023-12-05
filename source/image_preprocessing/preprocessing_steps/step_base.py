from abc import ABC, abstractmethod
from copy import deepcopy
import tensorflow as tf


from source.image_preprocessing.preprocessing_steps.step_utils import correct_image_tensor_shape

class StepBase(ABC):
    """  Base class for defining preprocessing steps for images in a image preprocessing pipeline.

    This abstract class provides a structured approach to implementing various image preprocessing steps. 
    Each step is characterized by its unique parameters and functionality, which are defined in the child classes 
    inheriting from StepBase. The class facilitates the integration and execution of preprocessing steps within a 
    TensorFlow image processing pipeline.

    Child classes should implement the `process_step` method according to their specific processing requirements and
    specify the value of the class attributes: `arguments_datatype` and `name`.
    Decorators `_tensor_pyfunc_wrapper` and `_nparray_pyfunc_wrapper` are provided for flexibility in implementing 
    TensorFlow and Python functions, respectively.

    Public Class Attribute (read-ony):
    - defaults_output_datatypes (dict):  A dictionary containing the defaults values of the argument datatypes.
    - arguments_datatype (dict):  A dictionary containing the preprocessing step
    specific values of the argument datatypes. Defaults to defaults_output_datatypes.
    - name (str): The base identifier for the preprocessing step.

    Public Instance Attribute (read-ony):
    - parameters (dict):  A dictionary containing parameters needed for the preprocessing step.

    Public Methods:
    - process_step(image: tf.Tensor or np.array) -> tf.Tensor or np.array:
        To be implemented by the child class to define the specific preprocessing functionality. The image datatype depends
        on the function_decorator used, either `_nparray_pyfunc_wrapper`  for processing tensors  or `_tensor_pyfunc_wrapper`
        for processing np.arrays.


    Child Class Template:
        class StepTemplate(StepBase):

            arguments_datatype = <dictionary specifying the argument datatypes>
            name = <Preprocessing step identifier>
            def __init__(self, **processing_step_specific_args):
                super().__init__(locals())

            @StepBase._nparray_pyfunc_wrapper # or @StepBase._tensor_pyfunc_wrapper depending on use case.
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
    
    default_output_datatypes = {'image': tf.uint8, 'target': tf.int8} 
    # Child Classes have to overwrite this attributes
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
        self._parameters = self._extract_parameters(local_vars)
        self.output_datatypes = deepcopy(self.default_output_datatypes) # Can be overwritten by child classes
        
    @property
    def parameters(self):
        """The parameters property is read-only."""
        return self._parameters
    
    def __eq__(self, obj: 'StepBase') -> bool:
        return self.name.split('__')[0] == obj.name.split('__')[0] and self._parameters == obj.parameters
        
    def _extract_parameters(self, local_vars):
        """  Extracts parameters needed for the preprocessing step based on local variables. It considers if parameters should be randomized or extracted directly from `local_vars`."""

        excluded_parameters = ['self', '__class__']
        initialization_parameters =  {key: value for key, value in local_vars.items() if key not in excluded_parameters}

        return initialization_parameters
    
    def get_step_json_representation(self):
        """Returns strings that corresponds to JSON entry text to be added to a JSON file."""

        # Convert datatype of values of parameters to match JSON format
        conv_parameters = {}
        for key, value in self._parameters.items():
            if isinstance(value, tuple):
                value = list(value)
            conv_parameters[key] = [value]

        parameters_str = ',\n'.join([f'        "{k}": {str(v).replace("True", "true").replace("False", "false")}' for k, v in conv_parameters.items()])
        json_string = f'    "{self.name}": {{\n{parameters_str}\n    }}' 
        return json_string
    
    @staticmethod
    def _tensor_pyfunc_wrapper(function):
        """ 
        A decorator for mapping TensorFlow tensor-based functions onto a dataset using tf.py_function.
        This decorator is designed for preprocessing steps implemented as Python functions,
        where the input to the function is a TensorFlow tensor.
        """
        def tensor_to_py_function_wrapper(self, image_tensor, target_tensor):
            processed_image = function(self, image_tensor)
            processed_image = tf.cast(processed_image, dtype=self.output_datatypes['image'])
            return processed_image, target_tensor

        def dataset_map_function(self, image_dataset):
            return image_dataset.map(
                lambda img, tgt: tf.py_function(
                    func=lambda i, t: tensor_to_py_function_wrapper(self, i, t),
                    inp=[img, tgt],
                    Tout=(self.output_datatypes['image'], self.output_datatypes['target'])
                )
            )

        return dataset_map_function

    @staticmethod
    def _nparray_pyfunc_wrapper(function):
        """ 
        A decorator for mapping Python functions (processing NumPy arrays) onto a TensorFlow dataset using tf.py_function.
        This decorator is useful for preprocessing steps implemented in Python,
        converting TensorFlow tensors to NumPy arrays before processing.
        """
        def numpy_to_py_function_wrapper(self, image_tensor, target_tensor):
            image_nparray = image_tensor.numpy().astype('uint8')
            processed_image = function(self, image_nparray)
            processed_image = tf.convert_to_tensor(processed_image, dtype=self.output_datatypes['image'])
            processed_image = correct_image_tensor_shape(processed_image)
            return processed_image, target_tensor

        def py_function_dataset_map(self, image_dataset):
            return image_dataset.map(
                lambda img, tgt: tf.py_function(
                    func=lambda i, t: numpy_to_py_function_wrapper(self, i, t),
                    inp=[img, tgt],
                    Tout=(self.output_datatypes['image'], self.output_datatypes['target'])
                )
            )

        return py_function_dataset_map

    @abstractmethod    
    def process_step(self, image_tensor, tf_target):
        # Child class must implement this method.
        pass


if __name__ == '__main__':
    print(StepBase.default_output_datatypes)