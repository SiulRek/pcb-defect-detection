import cv2
import numpy as np

from source.preprocessing.helpers.for_steps.step_base import StepBase


class DilationFilter(StepBase):
    """ A preprocessing step that applies dilation to an image. """

    arguments_datatype = {"kernel_size": int, "iterations": int}
    name = "Dilation Filter"

    def __init__(self, kernel_size=3, iterations=1):
        """
        Initializes the DilationFilter object that can be integrated into an
        image preprocessing pipeline.

        Args:
            - kernel_size (int, optional): The size of the dilation kernel.
                Defaults to 3.
            - iterations (int, optional): The number of times the dilation
                operation is applied. Defaults to 1.
        """
        super().__init__(locals())

    @StepBase._nparray_pyfunc_wrapper
    def process_step(self, image_nparray):
        kernel = np.ones(
            (self.parameters["kernel_size"], self.parameters["kernel_size"]), np.uint8
        )
        dilated_image = cv2.dilate(
            image_nparray, kernel, iterations=self.parameters["iterations"]
        )
        return dilated_image


if __name__ == "__main__":
    step = DilationFilter()
    print(step.get_step_json_representation())
