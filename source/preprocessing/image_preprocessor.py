from copy import deepcopy

import tensorflow as tf

from source.data_handling.manipulation.pack_images_and_labels import (
    pack_images_and_labels,
)
from source.data_handling.manipulation.unpack_dataset import unpack_dataset
from source.preprocessing.helpers.for_preprocessor.get_pipeline_code_representation import (
    get_pipeline_code_representation,
)
from source.preprocessing.helpers.for_preprocessor.json_instances_serializer import (
    JSONInstancesSerializer,
)
from source.preprocessing.helpers.for_preprocessor.step_class_mapping import (
    STEP_CLASS_MAPPING,
)
from source.preprocessing.helpers.for_steps.step_base import StepBase


class ImagePreprocessor:
    """
    Manages and processes a pipeline of image preprocessing steps for PCB
    images.

    The ImagePreprocessor class encapsulates a sequence of preprocessing
    operations defined as steps. Each step is a discrete preprocessing action,
    such as noise reduction, normalization, etc., applied in sequence to an
    input dataset of images.

    Attributes:
        - pipeline (list of StepBase Child classes): Preprocessing steps to
            be executed.
        - serializer (ClassInstancesSerializer): Serializes/deserializes the
            pipeline to/from JSON.

    Methods:
        - pipeline: Returns the current pipeline.
        - set_default_datatype: Sets the default datatype for the pipeline
            steps.
        - set_pipe: Sets the preprocessing pipeline with a deep copy of
            provided steps.
        - pipe_append: Appends a new step to the pipeline, verifying it is a
            subclass of StepBase.
        - pipe_pop: Pops a step from the pipeline.
        - process: Applies each preprocessing step to the provided dataset.
        - save_pipe_to_json: Serializes the preprocessing pipeline to a JSON
            file.
        - load_pipe_from_json: Loads and reconstructs a preprocessing
            pipeline from a JSON file.
        - load_randomized_pipe_from_json: Loads a pipeline from JSON with
            randomized parameters.

    Notes:
        - The pipeline should only contain instances of classes that inherit
            from StepBase.
        - The `set_pipe` and `pipe_append` methods include type checks to
            enforce this.
        - The JSON serialization and deserialization methods handle the
            conversion and reconstruction of the pipeline steps, respectively.
        - The `process` method's behavior changes based on the
            `raise_step_process_exception` flag, allowing for flexible error
            handling during the preprocessing stage.
        - The datatype thoughout the pipeline is the default datatype,
            except if explicitly set in the step.
        - tf.uint8 is the only input datatype all pipeline steps can handle.
    """

    def __init__(self, raise_step_process_exception=True):
        """
        Initializes the ImagePreprocessor with an empty pipeline. The
        `raise_step_process_exception` flag determines whether exceptions during
        step processing are raised or logged.
        """
        self._pipeline = []
        self._serializer = None
        self._initialize_class_instance_serializer(STEP_CLASS_MAPPING)
        self._raise_step_process_exception = raise_step_process_exception
        self._occurred_exception_message = ""
        self.set_default_datatype(tf.uint8)

    def __eq__(self, other):
        if not isinstance(other, ImagePreprocessor):
            return False
        return self.pipeline == other.pipeline

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
        """
        Checks if `step_class_mapping` is a dictionary and mapps to subclasses
        of `StepBase`, if successfull instanciates the
        `ClassInstancesSerializer` for pipeline serialization and
        deserialization.
        """
        if not isinstance(step_class_mapping, dict):
            msg = f"'step_class_mapping' must be of type dict not {type(step_class_mapping)}."
            raise TypeError(msg)

        for mapped_class in step_class_mapping.values():
            if not issubclass(mapped_class, StepBase):
                msg = (
                    "At least one mapped class is not a class or subclass of StepBase."
                )
                raise ValueError(msg)
        self._serializer = JSONInstancesSerializer(step_class_mapping)

    def set_default_datatype(self, datatype):
        """
        Sets the default datatype for the pipeline steps.

        Args:
            - datatype: The default output datatype if the pipeline steps.
                Must be a TensorFlow datatype (e.g., tf.float32, tf.uint8).
        """
        StepBase.default_output_datatype = datatype

    def set_pipe(self, pipeline):
        """
        Sets the preprocessing pipeline with a deep copy of the provided steps,
        ensuring each step is an instance of a StepBase subclass.

        Args:
            - pipeline (list[StepBase]): List of preprocessing steps to be
                set in the pipeline.
        """
        for step in pipeline:
            if not isinstance(step, StepBase):
                raise ValueError(
                    f"Expecting a Child of StepBase, got {type(step)} instead."
                )
        self._pipeline = deepcopy(pipeline)

    def pipe_pop(self):
        """
        Pops the last step from the pipeline.

        Returns:
            - StepBase: The last step that was removed from the pipeline.
        """
        return self._pipeline.pop()

    def pipe_append(self, step):
        """
        Appends a new step to the pipeline, verifying that it is a subclass of
        StepBase.

        Args:
            - step (StepBase): The preprocessing step to be appended to the
                pipeline.
        """
        if not isinstance(step, StepBase):
            raise ValueError(
                f"Expecting a Child of StepBase, got {type(step)} instead."
            )
        self._pipeline.append(deepcopy(step))

    def pipe_clear(self):
        """ Clears all steps from the pipeline """
        self._pipeline.clear()

    def save_pipe_to_json(self, json_path):
        """
        Serializes the preprocessing pipeline to the specified JSON file, saving
        the step hyperparameter configurations.

        Args:
            - json_path (str): File path where the pipeline configuration
                will be saved.
        """
        self.serializer.save_instances_to_json(self.pipeline, json_path)

    def load_pipe_from_json(self, json_path):
        """
        Loads and reconstructs a preprocessing pipeline from the specified JSON
        file.

        Args:
            - json_path (str): File path from where the pipeline
                configuration will be loaded.
        """
        self._pipeline = self.serializer.get_instances_from_json(json_path)

    def load_randomized_pipe_from_json(self, json_path):
        """
        Loads and reconstructs a preprocessing pipeline from the specified JSON
        file. The parameters of preprocessing steps are randomized from the
        specified range in the JSON file.

        Args:
            - json_path (str): File path from where the pipeline
                configuration will be loaded.
        """
        self._pipeline = self.serializer.get_randomized_instances_from_json(json_path)

    def get_pipe_code_representation(self):
        """
        Generates a text representation of the pipeline's configuration.

        Returns:
            - str: A string representation of the pipeline in a code-like
                format.
        """
        return get_pipeline_code_representation(self.pipeline)

    def _consume_tf_dataset(self, tf_dataset):
        """ Consumes a TensorFlow dataset to force the execution of the computation
        graph. """
        for _ in tf_dataset.take(1):
            pass

    def _strip_dataset(self, dataset):
        for element in dataset.take(1):
            if isinstance(element, tuple) and len(element) == 2:
                return unpack_dataset(dataset)
            return dataset, None

    def process(self, image_dataset):
        """
        Applies each preprocessing step to the provided dataset and returns the
        processed dataset. If `_raise_step_process_exception` is True,
        exceptions in processing a step will be caught and logged, and the
        process will return None. If False, it will proceed without exception
        handling.

        Args:
            - image_dataset (tf.data.Dataset): The TensorFlow dataset to be
                processed. It can contain images only or images and labels.

        Returns:
            - tf.data.Dataset: The processed dataset after applying all the
                steps in the pipeline.
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
                    msg = f"An error occurred in step {step.name}: {str(e)}"
                    print(msg)
                    self._occurred_exception_message = msg
                    return None

        self._consume_tf_dataset(processed_dataset)

        if label_dataset is not None:
            processed_dataset = pack_images_and_labels(processed_dataset, label_dataset)

        return processed_dataset
