"""
This module dynamically creates and manages unittest classes for testing most of the image preprocessing steps. It utilizes the dynamic class creation in Python to generate test cases for different preprocessing steps. This module allows additionally some tests to be skipped based on configuration flags.

Note:
    DynamicTestStep: A dynamically generated test class for a specific image preprocessing step.
    CREATE_VISUAL_INSPECTION:  Flag to enable or disable tests for visual inspection of processed images.
    steps_data: Contains the tuples <(step_class, arguments, grayscale_only)> of the image preprocessing steps to be tested.
"""

import unittest
from unittest import skip

from source.image_preprocessing.preprocessing_steps import AdaptiveHistogramEqualization, GaussianBlurring, GlobalHistogramEqualization
from source.image_preprocessing.tests.single_step_test import TestSingleStep


CREATE_VISUAL_INSPECTION = False


def create_test_class_for_step(step_class, arguments, grayscale_only=False):
    class DynamicTestStep(TestSingleStep):
        StepClass = step_class
        params = arguments
        process_grayscale_only = grayscale_only
        visual_inspection = CREATE_VISUAL_INSPECTION

        if grayscale_only:
            @skip("Processing of RGB images not enabled")
            def test_process_rgb_images(self):
                pass  

        if not CREATE_VISUAL_INSPECTION:
            @skip("Visual inspection not enabled")
            def test_processed_image_visualization(self):
                pass

    name = step_class.name.replace(' ', '')
    DynamicTestStep.__name__ = f'Test{name}'

    return DynamicTestStep


steps_data = [
    (AdaptiveHistogramEqualization, {'clip_limit': 1.0, 'tile_gridsize': (5, 5)}),
    (GlobalHistogramEqualization, {}, True),
    (GaussianBlurring, {'kernel_size': (5,5), 'sigma': 2.0})
]

# Test Class creation
for step_data in steps_data:
    name = step_data[0].name.replace(' ','')
    globals()[f'Test{name}'] = create_test_class_for_step(*step_data)


if __name__ == '__main__':
    unittest.main()