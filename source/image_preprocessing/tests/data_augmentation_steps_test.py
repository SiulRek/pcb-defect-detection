"""
This module dynamically creates and manages unittest classes for testing various data augmentation steps, with each class inheriting from `TestDataAugmentationStep`. 
It utilizes dynamic class creation in Python to generate test cases for different data augmentation techniques. 
This module allows some tests to be skipped based on configuration flags and focuses on evaluating the augmentation's effect on the image.

Key Components:
    - SingleStepTest: A class that dynamically generates test cases for each preprocessing step in the framework.
    - ENABLE_VISUAL_INSPECTION: A flag used to enable or disable tests that require visual inspection of augmented images.
    - augmentation_steps_data: A collection of tuples (augmentation_class, arguments), representing different data augmentation steps and their parameters to be tested.

Note:
    - This module accommodates variations in data augmentation steps, recognizing that not all test cases in `TestDataAugmentationStep` are universally applicable. Certain augmentations may necessitate customized modifications to the standard test cases, reflecting the diverse nature of image augmentation challenges.
"""
import unittest
from unittest import skip

import tensorflow as tf

import source.image_preprocessing.preprocessing_steps as steps
from source.image_preprocessing.tests.single_step_test import TestSingleStep

ENABLE_VISUAL_INSPECTION = True


def create_test_class_for_augmentation_step(augmentation_class, arguments):

    class DynamicDataAugmentationTest(TestSingleStep):
        TestStep = augmentation_class
        parameters = arguments

        if not ENABLE_VISUAL_INSPECTION:
            @skip("Visual inspection not enabled")
            def test_processed_image_visualization(self):
                pass
        
        if isinstance(augmentation_class(), steps.RandomCropper):
            def _verify_image_shapes(self, processed_images, original_images, color_channel_expected):
                """ 
                Helper method to verify the image dimensions and color channels in a processed dataset.
                Compa
                """
                for processed_image in processed_images:
                    processed_data_shape = tuple(processed_image.shape[:2].as_list())
                    self.assertEqual(processed_data_shape, self.parameters['crop_size'], 'heights and/or widths are not equal.') 
                    self.assertEqual(color_channel_expected, processed_image.shape[2], 'Color channels are not equal.')     


        def test_process_execution(self):
            """ 
            Test to verify the execution of the data augmentation step.

            Tests if at least one image is processed by the augmentation step.
            """
            image_dataset = self.image_dataset
            processed_images = self.test_step.process_step(image_dataset)
            for _ in processed_images.take(1):  
                pass
            equal_flag = True
            for ori_img, prc_img in zip(image_dataset, processed_images):
                prc_img = tf.cast(prc_img, dtype=ori_img.dtype)
                if ori_img.shape != prc_img.shape:
                    equal_flag = False
                elif not tf.reduce_all(tf.equal(ori_img, prc_img)).numpy():
                    equal_flag = False
            self.assertFalse(equal_flag)

    name = augmentation_class.name.replace(' ', '')
    DynamicDataAugmentationTest.__name__ = f'Test{name}'

    return DynamicDataAugmentationTest


augmentation_steps_data = [
    # (steps.RandomRotator, {'angle_range': (-90, 90)}),
    # (steps.RandomFlipper, {'flip_direction': 'horizontal'}),
    # (steps.GaussianNoiseInjector, {'mean': 0.0, 'sigma': 0.2, 'apply_clipping': True}),
    # (steps.RandomColorJitterer, {'brightness': 0.3, 'contrast': 0.3, 'saturation': 0.3, 'hue': 0.1}),
    (steps.RandomCropper, {'crop_size': (250, 250)}),
]


def load_multiple_augmentation_tests():
    """
    Dynamically loads and aggregates individual test suites for multiple data augmentation steps into a unified test suite.
    
    This function iterates over a predefined list of data augmentation steps and their corresponding arguments. For each step, 
    it dynamically creates a test class using `create_test_class_for_augmentation_step` and then loads the test cases from these 
    classes into individual test suites. These suites are then combined into a single comprehensive test suite.
    
    Returns:
        unittest.TestSuite: A combined test suite that aggregates tests for multiple data augmentation step test classes.
    """
    test_suites = []
    loader = unittest.TestLoader()
    for step_data in augmentation_steps_data:
        test_class = create_test_class_for_augmentation_step(*step_data)
        test_suites.append(loader.loadTestsFromTestCase(test_class))
    test_suite = unittest.TestSuite(test_suites) 
    return test_suite


if __name__ == '__main__':
    """ Main execution block for running the loaded test suite."""
    runner = unittest.TextTestRunner()
    test_suites = load_multiple_augmentation_tests()
    runner.run(test_suites)
