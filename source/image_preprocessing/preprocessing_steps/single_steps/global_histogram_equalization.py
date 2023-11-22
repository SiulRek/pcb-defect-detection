import cv2

from source.image_preprocessing.preprocessing_steps.step_base import StepBase


class GlobalHistogramEqualizer(StepBase):
    """  A preprocessing step that applies Contrast Limited Global Histogram Equalizer to an image."""
    arguments_datatype = {}
    name = 'Global Histogram Equalizer'

    def __init__(self):
        """ Initializes the GlobalHistogramEqualizer object that can be integrated in an image preprocessing pipeline."""
        super().__init__(locals())

    @StepBase._py_function_decorator
    def process_step(self, image_nparray):
        channels = cv2.split(image_nparray)
        eq_channels = [cv2.equalizeHist(ch) for ch in channels]  
        eq_image = cv2.merge(eq_channels)
        return eq_image
    

if __name__ == '__main__':
    step = GlobalHistogramEqualizer()
    print(step.get_step_json_representation())
    


