import tensorflow as tf

from source.preprocessing.helpers.for_steps.step_base import StepBase


class RandomColorJitterer(StepBase):
    """
    A data augmentation step that randomly alters the brightness, contrast,
    saturation, and hue of an image tensor. Each attribute is adjusted within a
    specified range. For grayscale images, only brightness and contrast
    adjustments are applied, as saturation and hue changes are not applicable.
    """

    arguments_datatype = {
        "brightness": float,
        "contrast": float,
        "saturation": float,
        "hue": float,
        "seed": int,
    }
    name = "Random Color Jitterer"

    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, seed=42):
        """
        Initializes the RandomColorJitterer object for integration into an image
        preprocessing pipeline.

        Args:
            - brightness (float): Maximum delta for brightness adjustment.
                Must be non-negative.
            - contrast (float): Contrast factor range (lower, upper). Must
                be non-negative.
            - saturation (float): Saturation factor range (lower, upper).
                Must be non-negative.
            - hue (float): Maximum delta for hue adjustment. Must be in [0,
                0.5].
            - seed (int): An optional integer seed for random operations.
                Default is 42.
        """
        super().__init__(locals())

    @StepBase._tensor_pyfunc_wrapper
    def process_step(self, image_tensor):
        is_grayscale = tf.shape(image_tensor)[-1] == 1
        image_tensor = tf.image.random_brightness(
            image_tensor,
            max_delta=self.parameters["brightness"],
            seed=self.parameters["seed"],
        )
        image_tensor = tf.image.random_contrast(
            image_tensor,
            lower=1 - self.parameters["contrast"],
            upper=1 + self.parameters["contrast"],
            seed=self.parameters["seed"],
        )

        if not is_grayscale:
            image_tensor = tf.image.random_saturation(
                image_tensor,
                lower=1 - self.parameters["saturation"],
                upper=1 + self.parameters["saturation"],
                seed=self.parameters["seed"],
            )
            image_tensor = tf.image.random_hue(
                image_tensor,
                max_delta=self.parameters["hue"],
                seed=self.parameters["seed"],
            )

        return image_tensor


if __name__ == "__main__":
    step = RandomColorJitterer()
    print(step.get_step_json_representation())
