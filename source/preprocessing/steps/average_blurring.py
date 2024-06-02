import cv2
from source.preprocessing.helpers.for_steps.step_base import StepBase


class AverageBlurFilter(StepBase):
    """ A preprocessing step that applies average blur filter to an image. """

    arguments_datatype = {"kernel_size": (int, int)}
    name = "Average Blur Filter"

    def __init__(self, kernel_size=(8, 8)):
        """
        Initializes the `AverageBlurFilter` object that can be integrated in an
        image preprocessing pipeline.

        Args:
            - kernel_size ((int, int)): The size of the averaging kernel.
                Both values should be positive integers.
        """
        super().__init__(locals())

    @StepBase._nparray_pyfunc_wrapper
    def process_step(self, image_nparray):
        ksize = self.parameters["kernel_size"]
        blurred_image = cv2.blur(image_nparray, ksize)
        return blurred_image


if __name__ == "__main__":
    step = AverageBlurFilter()
    print(step.get_step_json_representation())
