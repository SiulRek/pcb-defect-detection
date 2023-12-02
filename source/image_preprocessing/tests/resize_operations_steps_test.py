"""This module provides test suites for the preprocessing steps performing resize
operations within the image preprocessing pipeline. This module addresses the unique testing requirements for 
resize operations steps, ensuring their correct integration into the pipeline.

Key features include:

Note:
Helper steps, RGBToGrayscale and GrayscaleToRGB, are modified to support tf.float16, facilitating the assessment of 
resize operations in diverse scenarios.
"""

import unittest
from unittest import skip

import tensorflow as tf

from source.image_preprocessing.image_preprocessor import ImagePreprocessor
from source.image_preprocessing.image_preprocessor import StepBase
from source.image_preprocessing.tests.single_step_test import TestSingleStep
from source.image_preprocessing.preprocessing_steps.step_utils import correct_image_tensor_shape
import source.image_preprocessing.preprocessing_steps as steps


ENABLE_VISUAL_INSPECTION = True


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
    

class TestSquareShapePadder(TestSingleStep):
    """ A test suite for the SquareShapePadder step in the image preprocessing pipeline.

    This class inherits from TestSingleStep and is specifically designed to test the
    SquareShapePadder step, which pads images to make them square. It includes tests
    to verify that images are correctly padded with the specified pixel value.

    Attributes:
        TestStep: The step class to be tested, in this case, steps.SquareShapePadder.
        parameters: A dictionary containing parameters for the SquareShapePadder step.
    """

    TestStep = steps.SquareShapePadder
    parameters = {'padding_pixel_value': 0}

    if not ENABLE_VISUAL_INSPECTION:
        @skip("Visual inspection not enabled")
        def test_processed_image_visualization(self):
            pass
    
    def _verify_image_shapes(self, processed_dataset, original_dataset, color_channel_expected):
        """ 
        Helper method to verify the image dimensions and color channels in a processed dataset.
        Compares the processed images to the original dataset to ensure correct height, width, 
        and color channel transformations.
        """
        for original_data, processed_data in zip(original_dataset, processed_dataset):
            self.assertEqual(processed_data[1], original_data[1], 'Targets are not equal.')  
            processed_data_shape = tuple(processed_data[0].shape[:2].as_list())
            self.assertEqual(color_channel_expected, processed_data[0].shape[2], 'Color channels are not equal.') 
            self.assertEqual(processed_data_shape[0], processed_data_shape[1], 'Heights and widths are not equal.')     


class TestShapeResizer(TestSingleStep):
    """
    A test suite for the ShapeResizer step in the image preprocessing pipeline.

    This class extends TestSingleStep and focuses on testing the ShapeResizer step,
    which resizes images to a desired shape. It includes tests for verifying the
    resize operation on both RGB and grayscale images.

    Attributes:
        TestStep: The step class to be tested, in this case, steps.ShapeResizer.
        parameters: A dictionary of parameters for the ShapeResizer step.
    """

    TestStep = steps.ShapeResizer
    parameters = {'desired_shape': (1900, 2100), 'resize_method': 'bilinear'}

   
    @skip("Visual inspection not enabled")
    def test_processed_image_visualization(self):
        pass
    
    def _verify_image_shapes(self, processed_dataset, original_dataset, color_channel_expected):
        """ 
        Helper method to verify the image dimensions and color channels in a processed dataset.
        Compares the processed images to the original dataset to ensure correct height, width, 
        and color channel transformations.
        """
        for original_data, processed_data in zip(original_dataset, processed_dataset):
            self.assertEqual(processed_data[1], original_data[1], 'Targets are not equal.')  
            processed_data_shape = tuple(processed_data[0].shape[:2].as_list())
            self.assertEqual(color_channel_expected, processed_data[0].shape[2], 'Color channels are not equal.') 
            self.assertEqual(self.parameters['desired_shape'][0], processed_data_shape[0], 'heights are not like desired.')     
            self.assertEqual(self.parameters['desired_shape'][1], processed_data_shape[1], 'widths are not like desired.') 

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


def load_resize_operations_steps_tests():
    """
    Dynamically loads and aggregates individual test suites for resize operations preprocessing steps into a unified test suite.

    This function iterates over a predefined list of image preprocessing steps for resize operations and their corresponding arguments. For each step, 
    it dynamically creates a test class using `create_test_class_for_step` and then loads the test cases from these classes into 
    individual test suites. These suites are then combined into a single comprehensive test suite.

    Returns:
        unittest.TestSuite: A combined test suite that aggregates tests for multiple image preprocessing step test classes.
    """
    loader = unittest.TestLoader()
    test_suites = []
    test_suites.append(loader.loadTestsFromTestCase(TestSquareShapePadder))
    test_suites.append(loader.loadTestsFromTestCase(TestSingleStep))
    test_suite = unittest.TestSuite(test_suites)  # Combine the suites
    return test_suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(load_resize_operations_steps_tests())
