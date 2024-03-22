from copy import deepcopy

from source.image_preprocessing.preprocessing_steps.step_base import StepBase
from source.image_preprocessing.preprocessing_steps.step_class_mapping import STEP_CLASS_MAPPING
from source.utils import ClassInstancesSerializer
from source.load_raw_data.unpack_tf_dataset import unpack_tf_dataset
from source.load_raw_data.pack_images_and_labels import pack_images_and_labels

class ImagePreprocessor:
    """
    Manages and processes a pipeline of image preprocessing steps for PCB images.

    The ImagePreprocessor class encapsulates a sequence of preprocessing operations
    defined as steps. Each step is a discrete preprocessing action, such as noise
    reduction, normalization, etc. The steps are applied in sequence to an input
    dataset of images.

    Attributes (read only):
        pipeline (list of StepBase Child classes): A list of preprocessing steps to be executed.
        serializer (ClassInstancesSerializer): Handles the serialization and deserialization of the pipeline to and from a JSON file.


    Methods:
        pipeline(self):
            Property that returns the current pipeline.

        set_pipe(self, pipeline: List[StepBase]):
            Sets the preprocessing pipeline with a deep copy of the provided steps,
            ensuring each step is an instance of a StepBase subclass.

        pipe_append(self, step: StepBase):
            appendes a new step to the pipeline, verifying that it is a subclass of StepBase.

        pipe_pop(self) -> StepBase:
            Pops a step from pipeline.
            
        process(self, image_dataset: tf.data.Dataset) -> tf.data.Dataset:
            Applies each preprocessing step to the provided dataset and returns the processed dataset.

        save_pipe_to_json(self, filepath: str):
            Serializes the preprocessing pipeline to a JSON file, saving the steps and hyperparameter configurations.

        load_pipe_from_json(self, filepath: str):
            Loads and reconstructs a preprocessing pipeline from a JSON file.

        load_randomized_pipe_from_json(self, filepath: str):
            Loads and reconstructs a preprocessing pipeline from a JSON file. 

    Notes:
        - The pipeline should only contain instances of classes that inherit from StepBase.
        - The `set_pipe` and `pipe_append` methods include type checks to enforce this.
        - The JSON serialization and deserialization methods (`save_pipe_to_json` and `load_pipe_from_json`)
          handle the conversion and reconstruction of the pipeline steps, respectively.
        - The `process` method's behavior changes based on the `raise_step_process_exception` flag,
                allowing for flexible error handling during the preprocessing stage.
    """

    def __init__(self, raise_step_process_exception=True): 
        """ 
        Initializes the ImagePreprocessor with an empty pipeline.
        The `raise_step_process_exception` flag determines whether exceptions
        during step processing are raised or logged.
        """
        self._pipeline = []
        self._serializer = None
        self._initialize_class_instance_serializer(STEP_CLASS_MAPPING)
        self._raise_step_process_exception = raise_step_process_exception
        self._occurred_exception_message = ''

    @property
    def pipeline(self):
        return self._pipeline
    
    @property
    def serializer(self):
        return self._serializer
    
    @property
    def occurred_exception_message(self):
        return self._occurred_exception_message
    
    def _initialize_class_instance_serializer(self, step_class_mapping):
        """ Checks if `step_class_mapping` is a dictionary and mapps to subclasses of `StepBase`, if successfull instanciates the `ClassInstancesSerializer` for pipeline serialization and deserialization."""

        if not isinstance(step_class_mapping, dict):
            raise TypeError(f"'step_class_mapping' must be of type dict not {type(step_class_mapping)}.")
        
        for mapped_class in step_class_mapping.values():
            if not issubclass(mapped_class, StepBase):
                raise ValueError("At least one mapped class is not a class or subclass of StepBase.")
        self._serializer = ClassInstancesSerializer(step_class_mapping)

    def set_pipe(self, pipeline):
        """
        Sets the preprocessing pipeline with a deep copy of the provided steps,
        ensuring each step is an instance of a StepBase subclass.

        Args:
            pipeline (list[StepBase]): List of preprocessing steps to be set in the pipeline.
        """
        for step in pipeline:
            if not isinstance(step, StepBase):  
                raise ValueError(f'Expecting a Child of StepBase, got {type(step)} instead.')        
        self._pipeline = deepcopy(pipeline)

    def pipe_pop(self):
        """ Pops the last step from the pipeline.

        Returns:
            StepBase: The last step that was removed from the pipeline.
        """
        return self._pipeline.pop()

    def pipe_append(self, step):
        """
        Appends a new step to the pipeline, verifying that it is a subclass of StepBase.

        Args:
            step (StepBase): The preprocessing step to be appended to the pipeline.
        """
        if not isinstance(step, StepBase):  
                    raise ValueError(f'Expecting a Child of StepBase, got {type(step)} instead.')
        self._pipeline.append(deepcopy(step))
    
    def pipe_clear(self):
        """ Clears all steps from the pipeline"""
        self._pipeline.clear()

    def save_pipe_to_json(self, json_path):
        """
        Serializes the preprocessing pipeline to the specified JSON file, saving the step hyperparameter configurations.

        Args:
            json_path (str): File path where the pipeline configuration will be saved.
        """
        self.serializer.save_instances_to_json(self.pipeline, json_path)
        
    def load_pipe_from_json(self, json_path):
        """
        Loads and reconstructs a preprocessing pipeline from the specified JSON file.

        Args:
            json_path (str): File path from where the pipeline configuration will be loaded.
        """
        self._pipeline = self.serializer.get_instances_from_json(json_path)

    def load_randomized_pipe_from_json(self, json_path):
        """
        Loads and reconstructs a preprocessing pipeline from the specified JSON file.
        The parameters of preprocessing steps are randomized from the specified range in the JSON file.

        Args:
            json_path (str): File path from where the pipeline configuration will be loaded.
        """
        self._pipeline = self.serializer.get_randomized_instances_from_json(json_path)
        
    def get_pipe_code_representation(self):
        """
        Generates a text representation of the pipeline's configuration.

        Returns:
            str: A string representation of the pipeline in a code-like format.
        """
        
        if not self.pipeline:
            return "[]"

        repr = "[\n"
        for step in self.pipeline:
            q = "'" 
            parameters = ", ".join(f"{k}={q + v + q if isinstance(v, str) else v}" for k, v in step.parameters.items()) 
            step_repr = f"{step.__class__.__name__}({parameters})"
            repr += f"    {step_repr},\n"
        repr = repr[:-2] + "\n]"
        return repr

    def _consume_tf_dataset(self, tf_dataset):
        """
        Consumes a TensorFlow dataset to force the execution of the computation graph.
        """
        for _ in tf_dataset.take(1): 
            pass
    
    def _strip_dataset(self, dataset):
        for element in dataset.take(1):
            if isinstance(element, tuple) and len(element) == 2:
                return unpack_tf_dataset(dataset)
            return dataset, None

    def process(self, image_dataset):
        """
        Applies each preprocessing step to the provided dataset and returns the processed dataset.
        If `_raise_step_process_exception` is True, exceptions in processing a step will be caught and logged,
        and the process will return None. If False, it will proceed without exception handling.

        Args:
            image_dataset (tf.data.Dataset): The TensorFlow dataset to be processed. It can contain images only
            or images and labels.

        Returns:
            tf.data.Dataset: The processed dataset after applying all the steps in the pipeline.
        """
        image_dataset, label_dataset = self._strip_dataset(image_dataset)
        processed_dataset = image_dataset
        for step in self.pipeline:
            if self._raise_step_process_exception:
                processed_dataset = step.process_step(processed_dataset)
            else:
                try:
                    processed_dataset = step.process_step(processed_dataset)
                    self._consume_tf_dataset(processed_dataset)
                except Exception as e:
                    self._occurred_exception_message = f"An error occurred in step {step.name}: {str(e)}"
                    print(self._occurred_exception_message)
                    return None

        self._consume_tf_dataset(processed_dataset)

        if label_dataset is not None:
            processed_dataset = pack_images_and_labels(processed_dataset, label_dataset)

        return processed_dataset
    


