import cv2

from source.preprocessing.helpers.for_steps.step_base import StepBase


class AdaptiveThresholder(StepBase):
    """
    A preprocessing step that applies adaptive Thresholding to an image.

    Note: In the case of RGB images, it processes each color channel (Red, Green, Blue)
    separately.
    """

    arguments_datatype = {"block_size": int, "c": float}
    name = "Adaptive Thresholding"

    def __init__(self, block_size=15, c=-2):
        """
        Initializes the AdaptiveThresholder object that can be integrated in an image
            preprocessing pipeline.

        Args:
            block_size (int): Size of the pixel neighborhood that is used to calculate the threshold
                value.
            c (float): Constant subtracted from the mean or weighted mean.
        """
        super().__init__(locals())

    @StepBase._nparray_pyfunc_wrapper
    def process_step(self, image_nparray):

        def apply_adaptive_threshold(np_array):
            return cv2.adaptiveThreshold(
                np_array,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                blockSize=self.parameters["block_size"],  # Block size.
                C=self.parameters["c"],
            )

        if image_nparray.shape[2] == 1:
            thresholded_image = apply_adaptive_threshold(image_nparray)
        else:
            R, G, B = cv2.split(image_nparray)
            R_thresholded = apply_adaptive_threshold(R)
            G_thresholded = apply_adaptive_threshold(G)
            B_thresholded = apply_adaptive_threshold(B)
            thresholded_image = cv2.merge([R_thresholded, G_thresholded, B_thresholded])

        return thresholded_image


if __name__ == "__main__":
    step = AdaptiveThresholder()
    print(step.get_step_json_representation())
