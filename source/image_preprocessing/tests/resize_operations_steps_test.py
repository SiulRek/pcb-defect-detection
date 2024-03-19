"""This module provides test suites for the preprocessing steps performing resize
operations within the image preprocessing pipeline. This module addresses the unique testing requirements for 
resize operations steps, ensuring their correct integration into the pipeline.
"""

import unittest
from unittest import skip

from source.image_preprocessing.tests.single_step_test import TestSingleStep
import source.image_preprocessing.preprocessing_steps as steps


ENABLE_VISUAL_INSPECTION = False


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
        for original_image, processed_image in zip(original_dataset, processed_dataset):
            processed_image_shape = tuple(processed_image.shape[:2].as_list())
            original_image_shape = tuple(original_image.shape[:2].as_list())
            self.assertNotEqual(processed_image_shape, original_image_shape)
            self.assertEqual(color_channel_expected, processed_image.shape[2], 'Color channels are not equal.') 
            self.assertEqual(processed_image_shape[0], processed_image_shape[1], 'Heights and widths are not equal.')     


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
    parameters = {'desired_shape': (1900, 2100), 'resize_method': 'nearest'}

   
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
        for original_image, processed_image in zip(original_dataset, processed_dataset):
            processed_image_shape = tuple(processed_image.shape[:2].as_list())
            original_image_shape = tuple(original_image.shape[:2].as_list())
            self.assertNotEqual(processed_image_shape, original_image_shape)
            self.assertEqual(color_channel_expected, processed_image.shape[2], 'Color channels are not equal.') 
            self.assertEqual(self.parameters['desired_shape'][0], processed_image_shape[0], 'heights are not like desired.')     
            self.assertEqual(self.parameters['desired_shape'][1], processed_image_shape[1], 'widths are not like desired.') 


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
    test_suites.append(loader.loadTestsFromTestCase(TestShapeResizer))
    test_suite = unittest.TestSuite(test_suites)  # Combine the suites
    return test_suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(load_resize_operations_steps_tests())
