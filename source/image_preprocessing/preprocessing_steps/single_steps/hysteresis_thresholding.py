import cv2

from source.image_preprocessing.preprocessing_steps.step_base import StepBase


class HysteresisThresholder(StepBase):
    """ A preprocessing step that applies Hysteresis Thresholding to an image."""

    arguments_datatype = {'tresh': int}
    name = 'Hysteresis Thresholding'

    def __init__(self, tresh=128):
        """ Initializes the HysteresisThresholder object that can be integrated in an image preprocessing pipeline.
        
        Args:
            tresh (int, optional): The threshold value used for binary thresholding. Pixel values greater than 
                               this threshold are set to the maximum value (255, white), and values less than 
                               or equal to the threshold are set to 0 (black). Defaults to 128.

        """


        super().__init__(locals())

    @StepBase._py_function_decorator
    def process_step(self, image_nparray):
        
        def apply_hysteresis_treshold(np_array):
            _, thresholded_np_array= cv2.threshold(
                src=np_array, 
                thresh=self.params['tresh'], 
                maxval=255, 
                type=cv2.THRESH_BINARY
                )    
            return thresholded_np_array
        
        if image_nparray.shape[2] == 1:
            thresholded_image = apply_hysteresis_treshold(image_nparray)
        else:
            R, G, B = cv2.split(image_nparray)
            R_thresholded = apply_hysteresis_treshold(R)
            G_thresholded = apply_hysteresis_treshold(G)
            B_thresholded = apply_hysteresis_treshold(B)
            thresholded_image = cv2.merge([R_thresholded, G_thresholded, B_thresholded])

        return thresholded_image
    

if __name__ == '__main__':
    step = BinaryThresholder()
    print(step.get_step_json_representation())
    