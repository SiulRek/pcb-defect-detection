import cv2

from source.image_preprocessing.preprocessing_steps.step_base import StepBase


class GaussianBlurFilter(StepBase):
    """ A preprocessing step that applies Contrast Limited Gaussian Blur Filter to an image."""
    arguments_datatype = {'kernel_size': (int,int), 'sigma': float}
    name = 'Gaussian Blur Filter'

    def __init__(self, kernel_size=(5,5), sigma=0.3):
        """ 
        Initializes the GaussianBlurFilter object that can be integrated in an image preprocessing pipeline.
        
        Args:
            kernel_size ((int, int)): The size of the Gaussian kernel. Both values should be odd numbers.
            sigma (float): The standard deviation of the Gaussian kernel. A higher sigma means more blur.
        """
        super().__init__(locals())

    @StepBase._py_function_decorator
    def process_step(self, image_nparray):
        k = self.params['kernel_size']
        sigma = self.params['sigma']
        blurred_image= cv2.GaussianBlur(image_nparray, k, sigma)
        return blurred_image
    

if __name__ == '__main__':
    step = GaussianBlurFilter()
    print(step.get_step_json_representation)
    


