from tensorflow import image

from source.preprocessing.helpers.for_steps.step_base import StepBase


class GrayscaleToRGB(StepBase):
    """ A preprocessing step that converts Grayscaled images to RGB images. """

    arguments_datatype = {}
    name = "Grayscale To RGB"

    def __init__(self):
        """ Initializes the GrayscaleToRGB object that can be integrated in an image
        preprocessing pipeline. """
        super().__init__(locals())

    @StepBase._tensor_pyfunc_wrapper
    def process_step(self, image_tensor):
        if image_tensor.shape[2] == 1:
            processed_image = image.grayscale_to_rgb(image_tensor)
            return processed_image
        return image_tensor


if __name__ == "__main__":
    step = GrayscaleToRGB()
    print(step.get_step_json_representation())
