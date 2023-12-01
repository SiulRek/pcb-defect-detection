"""
This module comprises a suite of tests tailored for evaluating normalization steps within the 
image preprocessing pipeline. Unlike other preprocessing steps typically using tf.uint8 as the output 
datatype, normalization steps often use tf.float16. This module addresses the unique testing requirements for 
normalization steps, ensuring their correct integration into the pipeline.

Key features include:

Note:
Helper steps, RGBToGrayscale and GrayscaleToRGB, are modified to support tf.float16, facilitating the assessment of 
normalization steps in diverse scenarios.
"""

import unittest
from unittest import skip

import tensorflow as tf

from source.image_preprocessing.image_preprocessor import ImagePreprocessor
from source.image_preprocessing.image_preprocessor import StepBase
from source.image_preprocessing.tests.single_step_test import TestSingleStep
from source.image_preprocessing.preprocessing_steps.step_utils import correct_image_tensor_shape
import source.image_preprocessing.preprocessing_steps as steps


ENABLE_VISUAL_INSPECTION = False


class RGBToGrayscale(StepBase):
    arguments_datatype = {}
    name = 'RGB_to_Grayscale'

    def __init__(self):
        super().__init__(locals())

    @StepBase._tensor_pyfunc_wrapper
    def process_step(self, image_tensor):
        image_grayscale_tensor = tf.image.rgb_to_grayscale(image_tensor)
        image_grayscale_tensor = correct_image_tensor_shape(image_grayscale_tensor)
        return image_grayscale_tensor


class GrayscaleToRGB(StepBase):
    arguments_datatype = {}
    name = 'Grayscale_to_RGB'

    def __init__(self):
        super().__init__(locals())

    @StepBase._tensor_pyfunc_wrapper
    def process_step(self, image_tensor):
        image_tensor = tf.image.grayscale_to_rgb(image_tensor)
        image_grayscale_tensor = correct_image_tensor_shape(image_tensor)
        return image_grayscale_tensor


def create_test_class_for_step(step_class, arguments):

    class DynamicTestStep(TestSingleStep):
        TestStep = step_class
        parameters = arguments

        if not ENABLE_VISUAL_INSPECTION:
            @skip("Visual inspection not enabled")
            def test_processed_image_visualization(self):
                pass

        def test_process_rgb_images(self):
            """ 
            Test to ensure that RGB images are processed correctly. 
            Verifies that the RGB images, after processing, have the expected color channel dimensions.
            """
            rgb_to_grayscale_step = RGBToGrayscale()
            rgb_to_grayscale_step.output_datatypes['image'] = tf.float16
            pipeline = [self.test_step, rgb_to_grayscale_step]
            preprocessor = ImagePreprocessor()
            preprocessor.set_pipe(pipeline)
            processed_dataset = preprocessor.process(self.image_dataset)
            self._verify_image_shapes(processed_dataset, self.image_dataset, color_channel_expected=1)
        
        def test_process_grayscaled_images(self):
            """ 
            Test to ensure that grayscale images are processed correctly.
            Checks if the grayscale images maintain their dimensions after processing and 
            verifies the color channel transformation correctness.
            """
            pipeline = [RGBToGrayscale(), self.test_step]
            preprocessor = ImagePreprocessor()
            preprocessor.set_pipe(pipeline)
            processed_dataset = preprocessor.process(self.image_dataset)
            self._verify_image_shapes(processed_dataset, self.image_dataset, color_channel_expected=1)
            grayscale_to_rgb_step = GrayscaleToRGB()
            grayscale_to_rgb_step.output_datatypes['image'] = tf.float16
            processed_dataset = grayscale_to_rgb_step.process_step(processed_dataset)
            self._verify_image_shapes(processed_dataset, self.image_dataset, color_channel_expected=3)

    name = step_class.name.replace(' ', '')
    DynamicTestStep.__name__ = f'Test{name}'

    return DynamicTestStep


steps_data = [
    (steps.MinMaxNormalizer, {}),
    (steps.StandardNormalizer, {}),
    (steps.MeanNormalizer, {}),
    (steps.LocalContrastNormalizer, {'depth_radius': 5, 'bias': 1.0, 'alpha': 0.0001, 'beta': 0.75}),
]


def load_normalization_steps_tests():
    """

    Dynamically loads and aggregates individual test suites for normalize image preprocessing steps into a unified test suite.

    This function iterates over a predefined list of image preprocessing steps for normalization and their corresponding arguments. For each step, 
    it dynamically creates a test class using `create_test_class_for_step` and then loads the test cases from these classes into 
    individual test suites. These suites are then combined into a single comprehensive test suite.

    Returns:
        unittest.TestSuite: A combined test suite that aggregates tests for multiple image preprocessing step test classes.
    """
    test_suites = []
    loader = unittest.TestLoader()
    for step_data in steps_data:
        test_class = create_test_class_for_step(*step_data)
        test_suites.append(loader.loadTestsFromTestCase(test_class))
    test_suite = unittest.TestSuite(test_suites)  # Combine the suites
    return test_suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(load_normalization_steps_tests())
