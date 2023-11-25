"""This module contains tests for validating the functionality of the image preprocessing test framework and focuses on dynamically generated test classes. It provides tests to ensure that test classes are correctly created for one example preprocessing step and that conditional test logic (skipping certain tests) functions as expected."""

import unittest
from unittest.mock import patch

from source.image_preprocessing.preprocessing_steps.step_base import StepBase
from source.image_preprocessing.preprocessing_steps import AdaptiveHistogramEqualizer as ExampleStep
from source.image_preprocessing.tests.multiple_steps_test import create_test_class_for_step
from source.image_preprocessing.tests.single_step_test import TestSingleStep


class TestTestFramework(unittest.TestCase):
    def test_dynamic_class_creation(self):
        TestClass = create_test_class_for_step(ExampleStep, {'clip_limit': 1.0, 'tile_gridsize': (5, 5)})
        self.assertEqual(TestClass.__name__, 'TestAdaptiveHistogramEqualizer')
        self.assertTrue(issubclass(TestClass, TestSingleStep))
        self.assertTrue(hasattr(TestClass, 'StepClass'))
        self.assertTrue(hasattr(TestClass, 'params'))
        test_instance = TestClass('test_load_from_json')  # 'test_load_from_json' can be any existing test method name
        self.assertEqual(test_instance.StepClass, ExampleStep)
        self.assertEqual(test_instance.params,{'clip_limit': 1.0, 'tile_gridsize': (5, 5)})


class TestConditionalSkipping(unittest.TestCase):

    def test_visual_inspection_skipping(self):
        with patch('source.image_preprocessing.tests.multiple_steps_test.ENABLE_VISUAL_INSPECTION', False):
            TestClass = create_test_class_for_step(ExampleStep, {'clip_limit': 1.0, 'tile_gridsize': (5, 5)}, grayscale_only=True)
        suite = unittest.TestLoader().loadTestsFromTestCase(TestClass)
        result = unittest.TextTestRunner().run(suite)
        skipped_test_names = [test[0].id().split('.')[-1] for test in result.skipped]
        self.assertIn('test_processed_image_visualization', skipped_test_names)
        TestClass = create_test_class_for_step(ExampleStep, {})

    def test_grayscale_only_skipping(self):
        TestClass = create_test_class_for_step(ExampleStep, {'clip_limit': 1.0, 'tile_gridsize': (5, 5)}, grayscale_only=True)
        suite = unittest.TestLoader().loadTestsFromTestCase(TestClass)
        result = unittest.TextTestRunner().run(suite)
        skipped_test_names = [test[0].id().split('.')[-1] for test in result.skipped]
        self.assertIn('test_process_rgb_images', skipped_test_names)

    def test_visual_inspection_not_skipping(self):
        with patch('source.image_preprocessing.tests.multiple_steps_test.ENABLE_VISUAL_INSPECTION', True):
            TestClass = create_test_class_for_step(ExampleStep, {'clip_limit': 1.0, 'tile_gridsize': (5, 5)}, grayscale_only=True)
        suite = unittest.TestLoader().loadTestsFromTestCase(TestClass)
        result = unittest.TextTestRunner().run(suite)
        skipped_test_names = [test[0].id().split('.')[-1] for test in result.skipped]
        self.assertNotIn('test_processed_image_visualization', skipped_test_names)
        TestClass = create_test_class_for_step(ExampleStep, {})

    def test_grayscale_only_not_skipping(self):
        TestClass = create_test_class_for_step(ExampleStep, {'clip_limit': 1.0, 'tile_gridsize': (5, 5)}, grayscale_only=False)
        suite = unittest.TestLoader().loadTestsFromTestCase(TestClass)
        result = unittest.TextTestRunner().run(suite)
        skipped_test_names = [test[0].id().split('.')[-1] for test in result.skipped]
        self.assertNotIn('test_process_rgb_images', skipped_test_names)

if __name__ == '__main__':
    unittest.main()