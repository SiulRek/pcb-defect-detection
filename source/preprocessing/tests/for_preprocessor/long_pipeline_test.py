"""
This module contains tests for long image preprocessing pipelines.

The task is to build large pipelines (not necessarily the most efficient ones)
and test them on a dataset. The purpose of these long pipeline tests is to apply
a preconfigured image preprocessing pipeline to a dataset and verify if any
issues or bugs occur during the processing. This helps build confidence in the
framework's ability to handle complex pipelines without any issues.
"""

import os
import random
import unittest

import tensorflow as tf

from source.preprocessing.helpers.for_preprocessor.class_instances_serializer import (
    ClassInstancesSerializer,
)
from source.preprocessing.helpers.for_preprocessor.step_class_mapping import (
    STEP_CLASS_MAPPING,
)
from source.preprocessing.helpers.for_steps.step_base import StepBase
from source.preprocessing.helpers.for_tests.copy_json_exclude_entries import (
    copy_json_exclude_entries,
)
from source.preprocessing.image_preprocessor import ImagePreprocessor
from source.testing.base_test_case import BaseTestCase

N = 100  # Number of Pipelines Tests to run.


class RGBToGrayscale(StepBase):
    """ A preprocessing step that converts RGB image to Grayscale image. """

    arguments_datatype = {}
    name = "RGB To Grayscale"

    def __init__(self):
        """ Initializes the RGBToGrayscale object that can be integrated in an image
        preprocessing pipeline. """
        super().__init__(locals())

    @StepBase._tensor_pyfunc_wrapper
    def process_step(self, image_tensor):
        if image_tensor.shape[2] == 3:
            processed_image = tf.image.rgb_to_grayscale(image_tensor)
            return processed_image
        return image_tensor


class TestLongPipeline(BaseTestCase):
    """
    This class is the base class for all the long pipeline tests.

    The goal of the long pipeline tests is to identify if issues occure or bugs
    appear during processing of long pipelines.

    The class creates an image preprocessor with a preconfigured pipeline. The
    pipeline is constructed from a JSON template that contains all the image
    preprocessing steps of the framewor (some exclusions). The steps in the
    pipeline are shuffled randomly. The class includes a method called
    test_process_pipeline which applies the preprocessor to a dataset of images
    and verifies the output. If successful, it attempts to convert the processed
    dataset to grayscale and verifies the conversion as well. If the dataset is
    accurately converted to grayscale, it can be inferred that the pipeline has
    processed the dataset without any issues.
    """

    pipeline_id = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.json_template_file = os.path.join(
            cls.root_dir, r"source/preprocessing/pipelines/template.json"
        )
        cls.json_test_file = os.path.join(cls.temp_dir, "test_pipe.json")
        cls.log_file = os.path.join(cls.output_dir, "test_results.log")

        cls.image_dataset = cls.load_geometrical_forms_dataset()

    def setUp(self):
        super().setUp()
        with open(self.json_test_file, "a"):
            pass
        self.preprocessor = self.create_preconfigured_image_preprocessor()
        # print(self.preprocessor.get_pipe_code_representation())

    def create_preconfigured_image_preprocessor(self):
        excluded_keys = [
            "Non Local Mean Denoiser",
            "RGB To Grayscale",
            "Grayscale To RGB",
            "Local Contrast Normalizer",
            "Type Caster",
            "Random Color Jitterer",
        ]
        copy_json_exclude_entries(
            self.json_template_file, self.json_test_file, excluded_keys
        )

        serializer = ClassInstancesSerializer(STEP_CLASS_MAPPING)
        pipeline = serializer.get_instances_from_json(self.json_test_file)
        random.shuffle(pipeline)

        preprocessor = ImagePreprocessor()
        preprocessor.set_pipe(pipeline)
        return preprocessor

    def _verify_image_shapes(
        self, processed_dataset, original_dataset, color_channel_expected
    ):
        for original_image, processed_image in zip(original_dataset, processed_dataset):
            original_image = tf.cast(original_image, processed_image.dtype)
            if original_image.shape == processed_image.shape:
                if tf.reduce_all(tf.math.equal(original_image, processed_image)):
                    return False
            if (
                processed_image.shape[0] != processed_image.shape[1]
            ):  # Check if height and width are equal in processed data.
                return False
            if color_channel_expected != processed_image.shape[2]:
                return False
        return True

    def test_process_pipeline(self):
        try:
            processed_dataset = self.preprocessor.process(self.image_dataset)
        except Exception as e:
            raise BrokenPipeError(
                "An exception occurred while processing the dataset. This is the problematic pipeline: \n"
                + self.preprocessor.get_pipe_code_representation()
            ) from e

        if not self._verify_image_shapes(
            processed_dataset, self.image_dataset, color_channel_expected=3
        ):
            message = (
                "The processed dataset has unexpected shapes. This is the problematic pipeline: \n"
                + self.preprocessor.get_pipe_code_representation()
            )
            self.fail(message)

        grayscaled_dataset = RGBToGrayscale().process_step(processed_dataset)
        if not self._verify_image_shapes(
            grayscaled_dataset, self.image_dataset, color_channel_expected=1
        ):
            message = (
                "The processed dataset could not be converted to grayscale correctly. This is the problematic pipeline: \n"
                + self.preprocessor.get_pipe_code_representation()
            )
            self.fail(message)


def load_long_pipeline_tests(n=N):
    """
    This function dynamically creates n classes inheriting from TestLongPipeline
    and loads them into a test suite.

    Args:
        - n (int): Number of TestLongPipeline classes to create. Default to
            N.
    """

    test_suites = []
    for i in range(1, n + 1):
        test_class_name = f"{TestLongPipeline.__name__}_{i}"
        test_class = type(test_class_name, (TestLongPipeline,), {"pipeline_id": i})
        test_suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suites.append(test_suite)

    test_suite = unittest.TestSuite(test_suites)  # Combine the suites

    return test_suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(load_long_pipeline_tests(N))
