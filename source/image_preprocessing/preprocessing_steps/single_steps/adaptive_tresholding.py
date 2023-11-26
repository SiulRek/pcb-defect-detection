import cv2

from source.image_preprocessing.preprocessing_steps.step_base import StepBase


class AdaptiveTresholder(StepBase):
    """ A preprocessing step that applies adaptive Tresholding to an image."""

    arguments_datatype = {'block_size': int, 'c': float}
    name = 'Adaptive Tresholding'

    def __init__(self, block_size=15, c=-2):
        """ Initializes the AdaptiveTresholder object that can be integrated in an image preprocessing pipeline."""
        super().__init__(locals())

    @StepBase._py_function_decorator
    def process_step(self, image_nparray):
        
        def apply_adaptive_threshold(np_array):
            return cv2.adaptiveThreshold(np_array, 
                                         255, 
                                         cv2.ADAPTIVE_THRESH_MEAN_C, 
                                         cv2.THRESH_BINARY,
                                         blockSize=self.params['block_size'], # Block size.
                                         C=self.params['c']) 
        
        if image_nparray.shape[2] == 1:
            thresholded_image = apply_adaptive_threshold(image_nparray)
        else:
            R, G, B = cv2.split(image_nparray)
            R_thresholded = apply_adaptive_threshold(R)
            G_thresholded = apply_adaptive_threshold(G)
            B_thresholded = apply_adaptive_threshold(B)
            thresholded_image = cv2.merge([R_thresholded, G_thresholded, B_thresholded])

        return thresholded_image
    

if __name__ == '__main__':
    step = AdaptiveTresholder()
    print(step.get_step_json_representation())
    