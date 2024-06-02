import cv2

from source.preprocessing.helpers.for_steps.step_base import StepBase


class BilateralFilter(StepBase):
    """ A preprocessing step that applies bilateral filter to an image. """

    arguments_datatype = {"diameter": int, "sigma_color": float, "sigma_space": float}
    name = "Bilateral Filter"

    def __init__(self, diameter=9, sigma_color=75, sigma_space=75):
        """
        Initializes the `BilateralFilter` object that can be integrated in an
        image preprocessing pipeline.

        Args:
            - kernel_size ((int, int)): The size of the Gaussian kernel.
                Values should be odd numbers.
            - sigma (float): The standard deviation of the Gaussian kernel.
                A higher sigma means more blur.
        """
        super().__init__(locals())

    @StepBase._nparray_pyfunc_wrapper
    def process_step(self, image_nparray):
        blurred_image = cv2.bilateralFilter(
            src=image_nparray,
            d=self.parameters["diameter"],
            sigmaColor=self.parameters["sigma_color"],
            sigmaSpace=self.parameters["sigma_space"],
        )
        return blurred_image


if __name__ == "__main__":
    step = BilateralFilter()
    print(step.get_step_json_representation())
