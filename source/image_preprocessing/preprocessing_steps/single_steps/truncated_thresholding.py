import cv2

from source.image_preprocessing.preprocessing_steps.step_base import StepBase


class TruncatedThresholder(StepBase):
    """ A preprocessing step that applies truncated thresholding to an image. """

    arguments_datatype = {'thresh': int}
    name = 'Truncated Thresholding'

    def __init__(self, thresh=128):
        """ 
        Initializes the TruncatedThresholder object that can be integrated into an image preprocessing pipeline.

        Args:
            thresh (int, optional): The threshold value used for truncated thresholding. Pixel values greater than 
                                    this threshold are set to the threshold value itself, and values less than 
                                    or equal to the threshold remain unchanged. Defaults to 128.
        """
        super().__init__(locals())

    @StepBase._py_function_decorator
    def process_step(self, image_nparray):
        
        def apply_truncated_threshold(np_array):
            _, thresholded_np_array = cv2.threshold(
                src=np_array, 
                thresh=self.params['thresh'], 
                maxval=255, 
                type=cv2.THRESH_TRUNC
            )    
            return thresholded_np_array
        
        if image_nparray.shape[2] == 1:
            thresholded_image = apply_truncated_threshold(image_nparray)
        else:
            R, G, B = cv2.split(image_nparray)
            R_thresholded = apply_truncated_threshold(R)
            G_thresholded = apply_truncated_threshold(G)
            B_thresholded = apply_truncated_threshold(B)
            thresholded_image = cv2.merge([R_thresholded, G_thresholded, B_thresholded])

        return thresholded_image
    

if __name__ == '__main__':
    step = TruncatedThresholder()
    print(step.get_step_json_representation())
