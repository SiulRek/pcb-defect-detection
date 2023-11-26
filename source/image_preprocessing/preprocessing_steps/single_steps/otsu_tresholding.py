import cv2

from source.image_preprocessing.preprocessing_steps.step_base import StepBase


class OstuTresholder(StepBase):
    """ A preprocessing step that applies Otsu's Tresholding to an image."""

    arguments_datatype = {}
    name = 'Otsu Tresholding'

    def __init__(self):
        """ Initializes the OstuTresholder object that can be integrated in an image preprocessing pipeline."""
        super().__init__(locals())

    @StepBase._py_function_decorator
    def process_step(self, image_nparray):
        
        if image_nparray.shape[2] == 1:
            _, thresholded_image = cv2.threshold(image_nparray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)    
        else:
            R, G, B = cv2.split(image_nparray)
            _, R_thresholded = cv2.threshold(R, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, G_thresholded = cv2.threshold(G, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, B_thresholded = cv2.threshold(B, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresholded_image = cv2.merge([R_thresholded, G_thresholded, B_thresholded])

        return thresholded_image
    

if __name__ == '__main__':
    step = OstuTresholder()
    print(step.get_step_json_representation())
    