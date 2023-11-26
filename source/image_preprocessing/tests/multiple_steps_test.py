"""
This module dynamically creates and manages unittest classes for testing variuous of the image preprocessing steps. 
It utilizes the dynamic class creation in Python to generate test cases for different preprocessing steps. 
This module allows additionally some tests to be skipped based on configuration flags.

Key Components:
    - DynamicTestStep: A class that dynamically generates test cases for each specific image preprocessing step.
    - ENABLE_VISUAL_INSPECTION: A flag used to enable or disable tests that require visual inspection of processed images.
    - steps_data: A collection of tuples (step_class, arguments, grayscale_only), representing the different image preprocessing steps and their parameters to be tested.

Note:
    - The module accommodates variations in preprocessing steps, recognizing that not all test cases in `TestSingleStep` are universally applicable. Certain steps may necessitate customized modifications to the standard test cases, reflecting the diverse nature of image preprocessing challenges.
"""
import unittest
from unittest import skip

import source.image_preprocessing.preprocessing_steps as steps
from source.image_preprocessing.tests.single_step_test import TestSingleStep


ENABLE_VISUAL_INSPECTION = False


def create_test_class_for_step(step_class, arguments, grayscale_only=False):

    class DynamicTestStep(TestSingleStep):
        TestStep = step_class
        params = arguments
        process_grayscale_only = grayscale_only

        if grayscale_only:
            @skip("Processing of RGB images not enabled")
            def test_process_rgb_images(self):
                pass  

        if not ENABLE_VISUAL_INSPECTION:
            @skip("Visual inspection not enabled")
            def test_processed_image_visualization(self):
                pass

    name = step_class.name.replace(' ', '')
    DynamicTestStep.__name__ = f'Test{name}'

    return DynamicTestStep


steps_data = [
    (steps.AdaptiveHistogramEqualizer, {'clip_limit': 1.0, 'tile_gridsize': (5, 5)}),
    (steps.GlobalHistogramEqualizer, {}),
    (steps.GaussianBlurFilter, {'kernel_size': (5,5), 'sigma': 2.0}),
    (steps.MedianBlurFilter, {'kernel_size': 5}),
    (steps.BilateralFilter, {'diameter': 9, 'sigma_color':75, 'sigma_space':75}),
    (steps.AverageBlurFilter, {'kernel_size': (8,8)}),
    (steps.OstuTresholder, {}),
    (steps.AdaptiveTresholder,  {'block_size': 15, 'c':-2}),
]


def load_multiple_step_tests():
    """

    Dynamically loads and aggregates individual test suites for multiple image preprocessing steps into a unified test suite.

    This function iterates over a predefined list of image preprocessing steps and their corresponding arguments. For each step, 
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
    """ Main execution block for running the loaded test suite."""
    runner = unittest.TextTestRunner()
    runner.run(load_multiple_step_tests())
