import cv2

from source.preprocessing.helpers.for_steps.step_base import StepBase


class BinaryThresholder(StepBase):
    """
    A preprocessing step that applies binary Thresholding to an image.

    Note: In the case of RGB images, it processes each color channel (Red, Green, Blue)
    separately.
    """
    arguments_datatype = {'thresh': int}
    name = 'Binary Thresholding'

    def __init__(self, thresh=128):
        """ Initializes the BinaryThresholder object that can be integrated in an image
            preprocessing pipeline.

        Args:
            thresh (int, optional): The threshold value used for binary thresholding. Pixel values
                greater than this threshold are set to the maximum value (255, white), and values
                less than or equal to the threshold are set to 0 (black). Defaults to 128.

        """
        super().__init__(locals())

    @StepBase._nparray_pyfunc_wrapper
    def process_step(self, image_nparray):

        def apply_binary_threshold(np_array):
            _, thresholded_np_array= cv2.threshold(
                src=np_array,
                thresh=self.parameters['thresh'],
                maxval=255,
                type=cv2.THRESH_BINARY
                )
            return thresholded_np_array

        if image_nparray.shape[2] == 1:
            thresholded_image = apply_binary_threshold(image_nparray)
        else:
            R, G, B = cv2.split(image_nparray)
            r_thresholded = apply_binary_threshold(R)
            g_thresholded = apply_binary_threshold(G)
            b_thresholded = apply_binary_threshold(B)
            thresholded_image = cv2.merge([r_thresholded, g_thresholded, b_thresholded])

        return thresholded_image


if __name__ == '__main__':
    step = BinaryThresholder()
    print(step.get_step_json_representation())
