from abc import ABC, abstractmethod
import functools

from source.preprocessing.helpers.for_steps.get_step_json_representation import (
    get_step_json_representation,
)
from source.preprocessing.helpers.for_steps.step_utils import correct_image_tensor_shape
import tensorflow as tf


class StepBase(ABC):
    """
    Base class for defining preprocessing steps for images in an image
    preprocessing pipeline.

    This abstract class provides a structured approach to implementing various
    image preprocessing steps. Each step is characterized by its unique
    parameters and functionality, which are defined in the child classes
    inheriting from StepBase. The class facilitates the integration and
    execution of preprocessing steps within a TensorFlow image processing
    pipeline.

    Child classes should implement the `process_step` method according to their
    specific processing requirements and specify the value of the class
    attributes: `arguments_datatype` and `name`.

    Decorators `_tensor_pyfunc_wrapper` and `_nparray_pyfunc_wrapper` are
    provided for flexibility in implementing TensorFlow and Python functions,
    respectively.

    Public Class Attribute (read-only):
        - default_output_datatype (dtype): Default datatype for the output
            image tensor, can be overridden in child classes.
        - arguments_datatype (dtype, optional): Datatype for the
            preprocessing step's arguments. If not defined, defaults to
            `default_output_datatype`.
        - name (str): The base identifier for the preprocessing step.

    Public Instance Attribute (read-only):
        - parameters (dict): A dictionary containing parameters needed for
            the preprocessing step.

    Public Methods:
        - process_step(image_tensor: tf.Tensor, tf_target: Any) ->
        - tf.Tensor: To be implemented by the child class to define the
            specific preprocessing functionality. The method takes an image
            tensor and an optional target, returning the processed image tensor.

    Child Class Template:
        - class StepTemplate(StepBase): arguments_datatype = <datatype for
            arguments> name = <Preprocessing step identifier>
        - def __init__(self, **processing_step_specific_args):
            super().__init__(locals())

    @StepBase._nparray_pyfunc_wrapper # or @StepBase._tensor_pyfunc_wrapper def
    process_step(self, image_tensor): # TODO image_tensor_processed = ... return
    image_tensor_processed

    TODOs when integrating a new preprocessing step in the framework:
        - 1. Create preprocessing step class inheriting from `StepBase`
            according to the template. 2. Add mapping of the class to the
            constant `STEP_CLASS_MAPPING`
        - {<self.name>: type(self)}. 3. Add JSON entry of the class to
            .source/preprocessing/pipeline/template.json 4. Execute
            single_step_test.py over this class.
    """

    default_output_datatype = tf.uint8
    arguments_datatype = None  # Child Classes have to overwrite this attributes
    name = None

    def __init__(self, local_vars):
        """
        Constructs the base preprocessing step with a customizable name and set
        of parameters.

        This method serves as the foundational setup for all inherited
        preprocessing step classes. It integrates a unique identifier for each
        step and prepares the necessary parameters that dictate the behavior of
        the specific image preprocessing routine. It is designed to be flexible,
        allowing derived classes to pass in specific arguments that define the
        preprocessing step's unique characteristics and operational parameters.

        Args:
            - local_vars (dict): A collection of variables provided by the
                child class instantiation that includes hyperparameter
                configurations.
        """
        self._parameters = self._extract_parameters(local_vars)
        self.output_datatype = self.default_output_datatype

    @property
    def parameters(self):
        """ The parameters property is read-only. """
        return self._parameters

    def __eq__(self, obj: "StepBase") -> bool:
        return (
            self.name.split("__")[0] == obj.name.split("__")[0]
            and self._parameters == obj.parameters
        )

    def _extract_parameters(self, local_vars):
        """
        Extracts parameters needed for the preprocessing step based on local
        variables. It considers if parameters should be randomized or extracted
        directly from `local_vars`.
        """
        excluded_parameters = ["self", "__class__"]
        initialization_parameters = {
            key: value
            for key, value in local_vars.items()
            if key not in excluded_parameters
        }
        return initialization_parameters

    def get_step_json_representation(self):
        """
        Returns strings that corresponds to JSON entry text of the preprocessing
        step to be added to a JSON file.

        Returns:
            - str: The JSON representation of the preprocessing step.
        """
        return get_step_json_representation(self.parameters, self.name)

    @staticmethod
    def _tensor_pyfunc_wrapper(function):
        @functools.wraps(function)  # Preserve function metadata
        def tensor_to_py_function_wrapper(self, image_tensor):
            processed_image = function(self, image_tensor)
            processed_image = tf.cast(processed_image, dtype=self.output_datatype)
            return processed_image

        @functools.wraps(function)  # Preserve function metadata
        def dataset_map_function(self, image_dataset):
            return image_dataset.map(
                lambda img: tf.py_function(
                    func=lambda i: tensor_to_py_function_wrapper(self, i),
                    inp=[img],
                    Tout=(self.output_datatype),
                )
            )

        return dataset_map_function

    @staticmethod
    def _nparray_pyfunc_wrapper(function):
        @functools.wraps(function)  # Preserve function metadata
        def numpy_to_py_function_wrapper(self, image_tensor):
            image_nparray = image_tensor.numpy().astype("uint8")
            processed_image = function(self, image_nparray)
            processed_image = tf.convert_to_tensor(
                processed_image, dtype=self.output_datatype
            )
            processed_image = correct_image_tensor_shape(processed_image)
            return processed_image

        @functools.wraps(function)  # Preserve function metadata
        def py_function_dataset_map(self, image_dataset):
            return image_dataset.map(
                lambda img: tf.py_function(
                    func=lambda i: numpy_to_py_function_wrapper(self, i),
                    inp=[img],
                    Tout=(self.output_datatype),
                )
            )

        return py_function_dataset_map

    @abstractmethod
    def process_step(self, image_tensor):
        # Child class must implement this method.
        raise NotImplementedError


if __name__ == "__main__":
    print(StepBase.default_output_datatype)
