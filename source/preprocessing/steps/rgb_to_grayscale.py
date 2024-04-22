from tensorflow import image

from source.preprocessing.helpers.step_base import StepBase


class RGBToGrayscale(StepBase):
    """  A preprocessing step that converts RGB image to Grayscale image."""
    arguments_datatype = {}
    name = 'RGB To Grayscale'

    def __init__(self):
        """ Initializes the RGBToGrayscale object that can be integrated in an image preprocessing
        pipeline."""
        super().__init__(locals())

    @StepBase._tensor_pyfunc_wrapper
    def process_step(self, image_tensor):
        if image_tensor.shape[2] == 3:
            processed_image = image.rgb_to_grayscale(image_tensor)
            return processed_image
        return image_tensor


if __name__ == '__main__':
    step = RGBToGrayscale()
    print(step.get_step_json_representation())
