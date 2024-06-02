import tensorflow as tf

from source.preprocessing.helpers.for_steps.step_base import StepBase


class Mirrorer(StepBase):
    """ A preprocessing step that mirrors an image tensor either horizontally or
    vertically. The direction of mirroring is specified as an input parameter. """

    arguments_datatype = {"mirror_direction": str}
    name = "Mirrorer"

    def __init__(self, mirror_direction="horizontal"):
        """
        Initializes the Mirrorer object for integration into an image
        preprocessing pipeline.

        Args:
            - mirror_direction (str): The direction for mirroring the image.
                Accepts 'horizontal' or 'vertical'. Default is 'horizontal'.
        """
        super().__init__(locals())

    @StepBase._tensor_pyfunc_wrapper
    def process_step(self, image_tensor):
        if self.parameters["mirror_direction"] == "horizontal":
            mirrored_image = tf.image.flip_left_right(image_tensor)
        elif self.parameters["mirror_direction"] == "vertical":
            mirrored_image = tf.image.flip_up_down(image_tensor)
        else:
            raise ValueError(
                "Invalid mirror direction. Choose 'horizontal' or 'vertical'."
            )
        return mirrored_image


if __name__ == "__main__":
    step = Mirrorer()
    print(step.get_step_json_representation())
