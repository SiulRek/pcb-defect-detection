import unittest

import matplotlib.pyplot as plt

from source.testing.base_test_case import BaseTestCase


class BaseTestCaseDemo(BaseTestCase):
    """
    A demonstration class to prove and showcase the functionality of the
    BaseTestCase. It leverages the setup and teardown mechanisms of BaseTestCase
    to demonstrate their effectiveness and usage in a practical testing
    scenario.
    """

    @classmethod
    def compute_output_dir(cls):
        # Overriding the method to avoid the need for a 'tests' directory.
        return super().compute_output_dir("testing")

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        print(f"SetupClass: Output directory has been set up at {cls.output_dir}")
        print(f"SetupClass: Temporary directory has been set up at {cls.temp_dir}")
        print(f"SetupClass: Log file has been set up at {cls.log_file}")

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        print(
            f"TearDownClass: Temporary directory at {cls.temp_dir} has been cleaned up."
        )

    def test_example_functionality(self):
        """ An example test that logs its outcome and demonstrates the logging
        functionality. """
        self.assertTrue(True)

    def test_load_image_dataset(self):
        """ An example test that demonstrates the usage of a helper method. """
        dataset = self.load_image_dataset()
        for image in dataset.take(1):
            # plot the image to outputs directory
            self.assertIsNotNone(image)
            plt.imshow(image)
            plt.savefig(f"{self.output_dir}/loaded_image.png")

    def tearDown(self):
        super().tearDown()
        print("Logging the outcome of the test method.")


if __name__ == "__main__":

    unittest.main()
