import json
import re

from copy import deepcopy
import tensorflow as tf

from python_code.image_preprocessing.preprocessing_steps.step_base import StepBase
from python_code.image_preprocessing.preprocessing_steps.step_class_mapping import STEP_CLASS_MAPPING
from python_code.utils import ClassInstanceSerializer

class ImagePreprocessor:
    """
    Manages and processes a pipeline of image preprocessing steps for PCB images.

    The ImagePreprocessor class encapsulates a sequence of preprocessing operations
    defined as steps. Each step is a discrete preprocessing action, such as noise
    reduction, normalization, etc. The steps are applied in sequence to an input
    dataset of images.

    Attributes (read only):
        pipeline (list of StepBase Child classes): A list of preprocessing steps to be executed.
        class_instance_serializer (ClassInstanceSerializer): Handles the serialization and deserialization of the pipeline to and from a JSON file.


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
        self._class_instance_serializer = None
        self._initialize_class_instance_serializer(STEP_CLASS_MAPPING)
        self._raise_step_process_exception = raise_step_process_exception

    @property
    def pipeline(self):
        return self._pipeline
    
    @property
    def class_instance_serializer(self):
        return self._class_instance_serializer
    
    def _initialize_class_instance_serializer(self, step_class_mapping):
        """ Checks if `step_class_mapping` is a dictionary and mapps to subclasses of `StepBase`, if successfull instanciates the `ClassInstanceSerializer` for pipeline serialization and deserialization."""
        if not type(step_class_mapping) is dict:
            raise TypeError(f"'step_class_mapping' must be of type dict not {type(step_class_mapping)}.")
        else:
            for mapped_class in step_class_mapping.values():
                if not issubclass(mapped_class, StepBase):
                    raise ValueError("At least one mapped class is not a class or subclass of StepBase.")
            self._class_instance_serializer = ClassInstanceSerializer(step_class_mapping)

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
            if self._raise_step_process_exception:
                processed_dataset = step.process_step(processed_dataset)
            else:
                try:
                    processed_dataset = step.process_step(processed_dataset)
                    self._consume_tf_dataset(processed_dataset)
                except Exception as e:
                    print(f"An error occurred in step {step.name}: {str(e)}")
                    return None

        return processed_dataset
    
    def _consume_tf_dataset(self, tf_dataset):
        """
        Consumes a TensorFlow dataset to force the execution of the computation graph.
        """
        for _, _ in tf_dataset.take(1): 
            pass

    def save_pipe_to_json(self, json_path):
        "Serializes the preprocessing pipeline to the specified JSON file, saving the step configurations."
        if self.class_instance_serializer:
            self.class_instance_serializer.save_instance_list_to_json(self.pipeline, json_path)
        else:
            raise AttributeError(f"Not None Instance Attribute 'class_instance_serializer' is required to save pipe.")
    
        
    def load_pipe_from_json(self, json_path):
        """  Loads and reconstructs a preprocessing pipeline from the specified JSON file.
        """
        
        if self.class_instance_serializer:
             self._pipeline = self.class_instance_serializer.get_instance_list_from_json(json_path)
        else:
            raise AttributeError(f"Not None Instance Attribute 'class_instance_serializer' is required to save pipe.")


