"""This module provides test suites for the preprocessing steps performing resize
operations within the image preprocessing pipeline. """


import unittest
from unittest import skip

from source.image_preprocessing.preprocessing_steps import SquareShapePadder
from source.image_preprocessing.preprocessing_steps import ShapeResizer
from source.image_preprocessing.tests.single_step_test import TestSingleStep

ENABLE_VISUAL_INSPECTION = True

class TestSquareShapePadder(TestSingleStep):
    """
    A test suite for verifying the functionality of the SquareShapePadder preprocessing step.

    Inherits from TestSingleStep and focuses on validating that the SquareShapePadder step
    correctly pads images to a square shape without altering the image content.
    """

    params = {'padding_pixel_value': 0}
    TestStep = SquareShapePadder
    process_grayscale_only = False

    def _verify_image_shapes(self, processed_dataset, original_dataset, color_channel_expected):
        """
        Overridden helper method to add verification that the SquareShapePadder step correctly adjusts the image dimensions.
        """
        for original_data, processed_data in zip(original_dataset, processed_dataset):
            self.assertEqual(processed_data[1], original_data[1], 'Targets are not equal.')  
            self.assertEqual(processed_data[0].shape[0], processed_data[0].shape[1], 'height and width are not equal.') 
            self.assertEqual(color_channel_expected, processed_data[0].shape[2], 'Color channels are not equal.')   

    if not ENABLE_VISUAL_INSPECTION:
        @skip("Visual inspection not enabled")
        def test_processed_image_visualization(self):
            pass


class TestShapeResizer(TestSingleStep):
    """
    A test suite for verifying the functionality of the ShapeResizer preprocessing step.

    Inherits from TestSingleStep and focuses on validating that the ShapeResizer step
    correctly resizes images to the specified dimensions using the desired resize method. 
    It includes tests to ensure that both the aspect ratio changes and the resizing process 
    itself do not adversely affect the integrity of the image data.
    """

    params = {'desired_shape': (1900, 2100), 'resize_method': 'bilinear'}
    TestStep = ShapeResizer
    process_grayscale_only = False

    def _verify_image_shapes(self, processed_dataset, original_dataset, color_channel_expected):
        """
        Overridden helper method to verify that the ShapeResizer step correctly adjusts the image dimensions.

        Asserts that the processed images have the expected height and width as specified by the desired_shape
        parameter and that the color channels remain consistent post-processing.
        """
        height, width = self.params['desired_shape']
        for original_data, processed_data in zip(original_dataset, processed_dataset):
            self.assertEqual(processed_data[1], original_data[1], 'Targets are not equal.')  
            self.assertEqual(processed_data[0].shape[0], height, 'Heights are not equal.') 
            self.assertEqual(processed_data[0].shape[1], width, 'Widths are not equal.') 
            self.assertEqual(color_channel_expected, processed_data[0].shape[2], 'Color channels are not equal.')   

    if not ENABLE_VISUAL_INSPECTION:
        @skip("Visual inspection not enabled")
        def test_processed_image_visualization(self):
            pass


def load_resize_operations_tests():
    """
    Loads and combines test suites for resize operations preprocessing steps into a single test suite.  
    
    Returns:
        unittest.TestSuite: A test suite containing all test cases for the SquareShapePadder preprocessing step.
    """
    loader = unittest.TestLoader()
    test_suites = []
    test_suites.append(loader.loadTestsFromTestCase(TestSquareShapePadder))
    test_suites.append(loader.loadTestsFromTestCase(TestShapeResizer))
    return unittest.TestSuite(test_suites)


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(load_resize_operations_tests())
