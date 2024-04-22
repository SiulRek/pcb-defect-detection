import cv2

from source.preprocessing.helpers.for_steps.step_base import StepBase


class AdaptiveHistogramEqualizer(StepBase):
    """
    A preprocessing step that applies Contrast Limited Adaptive Histogram Equalizer (CLAHE) to an
    image.

    Note:     In the case of RGB images, it processes each color channel (Red, Green, Blue)
    separately.
    """
    arguments_datatype = {'clip_limit': float, 'tile_gridsize': (int, int)}
    name = 'Adaptive Histogram Equalizer'

    def __init__(self, clip_limit=2.0, tile_gridsize=(8,8)):
        """ Initializes the AdaptiveHistogramEqualizer object that can be integrated in an image
          preprocessing pipeline.

        Parameters:
            clip_limit (float): Threshold for contrast limiting. Higher values increase contrast;
                                too high values may lead to noise amplification.
            tile_gridsize (tuple): Size of the grid for the tiles (regions) of the image to which
                                CLAHE will be applied. Smaller tiles can lead to more localized
                                contrast enhancement.
        """
        super().__init__(locals())

    @StepBase._nparray_pyfunc_wrapper
    def process_step(self, image_nparray):
        channels = cv2.split(image_nparray)
        clahe = cv2.createCLAHE(clipLimit=self.parameters['clip_limit'],
                                    tileGridSize=self.parameters['tile_gridsize'])
        clahe_channels = [clahe.apply(ch) for ch in channels]
        clahe_image = cv2.merge(clahe_channels)
        return clahe_image


if __name__ == '__main__':
    step = AdaptiveHistogramEqualizer()
    print(step.get_step_json_representation())
