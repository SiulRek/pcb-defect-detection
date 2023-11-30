import cv2

from source.image_preprocessing.preprocessing_steps.step_base import StepBase


class NLMeanDenoiser(StepBase):
    """ A preprocessing step that applies Contrast Limited Adaptive Histogram Equalizer (CLAHE) to an image."""

    arguments_datatype = {'h': float, 'template_window_size': int, 'search_window_size': int}
    name = 'Non Local Mean Denoiser'

    def __init__(self, h=1.0, template_window_size=7, search_window_size=21):
        """
        Initializes the NLMeanDenoiser object for integration in an image preprocessing pipeline.

        Args:
            h (float): Filter strength. Higher values remove noise better but may also remove image details.
            template_window_size (int): Odd size of the window used to compute the weighted average for the given pixel.
            search_window_size (int): Odd size of the window used to search for patches similar to the one centered at the current pixel.
        """
        super().__init__(locals())

    @StepBase._nparray_pyfunc_wrapper
    def process_step(self, image_nparray):
        denoised_image = cv2.fastNlMeansDenoising(
                src=image_nparray,
                h=self.params['h'], 
                templateWindowSize=self.params['template_window_size'], 
                searchWindowSize=self.params['search_window_size'])
        return denoised_image
    

if __name__ == '__main__':
    step = NLMeanDenoiser()
    print(step.get_step_json_representation())
    



