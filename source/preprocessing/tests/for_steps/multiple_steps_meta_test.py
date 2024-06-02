"""
This module contains tests for validating the functionality of the image
preprocessing test framework and focuses on dynamically generated test classes.
It provides tests to ensure that test classes are correctly created for one
example preprocessing step and that conditional test logic (skipping certain
tests) functions as expected.
"""

import unittest
from unittest.mock import patch

from source.preprocessing.steps import AdaptiveHistogramEqualizer as ExampleStep
from source.preprocessing.tests.for_steps.single_step_test import TestSingleStep
from source.preprocessing.tests.multiple_steps_test import create_test_class_for_step


class TestTestFramework(unittest.TestCase):
    def test_dynamic_class_creation(self):
        TestClass = create_test_class_for_step(
            ExampleStep, {"clip_limit": 1.0, "tile_gridsize": (5, 5)}
        )
        self.assertEqual(TestClass.__name__, "TestAdaptiveHistogramEqualizer")
        self.assertTrue(issubclass(TestClass, TestSingleStep))
        self.assertTrue(hasattr(TestClass, "TestStep"))
        self.assertTrue(hasattr(TestClass, "parameters"))
        test_instance = TestClass(
            "test_load_from_json"
        )  # 'test_load_from_json' can be any existing test method name
        self.assertEqual(test_instance.TestStep, ExampleStep)
        self.assertEqual(
            test_instance.parameters, {"clip_limit": 1.0, "tile_gridsize": (5, 5)}
        )


class TestConditionalSkipping(unittest.TestCase):

    def test_visual_inspection_skipping_1(self):
        with patch(
            "source.preprocessing.tests.multiple_steps_test.ENABLE_VISUAL_INSPECTION",
            False,
        ):
            TestClass = create_test_class_for_step(
                ExampleStep, {"clip_limit": 1.0, "tile_gridsize": (5, 5)}
            )
        suite = unittest.TestLoader().loadTestsFromTestCase(TestClass)
        result = unittest.TextTestRunner().run(suite)
        skipped_test_names = [test[0].id().split(".")[-1] for test in result.skipped]
        self.assertIn("test_processed_image_visualization", skipped_test_names)
        TestClass = create_test_class_for_step(ExampleStep, {})

    def test_visual_inspection_skipping_2(self):
        with patch(
            "source.preprocessing.tests.multiple_steps_test.ENABLE_VISUAL_INSPECTION",
            True,
        ):
            TestClass = create_test_class_for_step(
                ExampleStep,
                {"clip_limit": 1.0, "tile_gridsize": (5, 5)},
                visual_inspection_always_disable=True,
            )
        suite = unittest.TestLoader().loadTestsFromTestCase(TestClass)
        result = unittest.TextTestRunner().run(suite)
        skipped_test_names = [test[0].id().split(".")[-1] for test in result.skipped]
        self.assertIn("test_processed_image_visualization", skipped_test_names)
        TestClass = create_test_class_for_step(ExampleStep, {})

    def test_visual_inspection_not_skipping(self):
        with patch(
            "source.preprocessing.tests.multiple_steps_test.ENABLE_VISUAL_INSPECTION",
            True,
        ):
            TestClass = create_test_class_for_step(
                ExampleStep, {"clip_limit": 1.0, "tile_gridsize": (5, 5)}
            )
        suite = unittest.TestLoader().loadTestsFromTestCase(TestClass)
        result = unittest.TextTestRunner().run(suite)
        skipped_test_names = [test[0].id().split(".")[-1] for test in result.skipped]
        self.assertNotIn("test_processed_image_visualization", skipped_test_names)
        TestClass = create_test_class_for_step(ExampleStep, {})


if __name__ == "__main__":
    unittest.main()
