import cv2

from source.image_preprocessing.preprocessing_steps.step_base import StepBase


class AverageBlurFilter(StepBase):
    """ A preprocessing step that applies average blur filter to an image."""
    arguments_datatype = {'kernel_size': (int,int)}
    name = 'Average Blur Filter'

    def __init__(self, kernel_size=(8,8)):
        """ Initializes the `AverageBlurFilter` object that can be integrated in an image preprocessing pipeline."""
        super().__init__(locals())

    @StepBase._py_function_decorator
    def process_step(self, image_nparray):
        ksize = self.params['kernel_size'] 
        blurred_image = cv2.blur(image_nparray, ksize)
        return blurred_image
    

if __name__ == '__main__':
    step = AverageBlurFilter()
    print(step.get_step_json_representation())
    


