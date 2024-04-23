import cv2

from source.preprocessing.helpers.for_steps.step_base import StepBase


class OstuThresholder(StepBase):
    """
    A preprocessing step that applies Otsu's Thresholding to an image.

    Note:     In the case of RGB images, it processes each color channel (Red, Green, Blue)
    separately.
    """

    arguments_datatype = {}
    name = "Otsu Thresholding"

    def __init__(self):
        """Initializes the OstuThresholder object that can be integrated in an image preprocessing
        pipeline."""
        super().__init__(locals())

    @StepBase._nparray_pyfunc_wrapper
    def process_step(self, image_nparray):

        if image_nparray.shape[2] == 1:
            _, thresholded_image = cv2.threshold(
                image_nparray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            R, G, B = cv2.split(image_nparray)
            _, r_thresholded = cv2.threshold(
                R, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            _, g_thresholded = cv2.threshold(
                G, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            _, b_thresholded = cv2.threshold(
                B, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            thresholded_image = cv2.merge([r_thresholded, g_thresholded, b_thresholded])

        return thresholded_image


if __name__ == "__main__":
    step = OstuThresholder()
    print(step.get_step_json_representation())
