from source.preprocessing.helpers.for_steps.step_base import StepBase
import tensorflow as tf


class RandomCropper(StepBase):
    """ A data augmentation step that randomly crops a portion of the image. """

    arguments_datatype = {"crop_size": (int, int), "seed": int}
    name = "Random Cropper"

    def __init__(self, crop_size=(256, 256), seed=42):
        """
        Initializes the RandomCropper object for integration into an image
        preprocessing pipeline.

        Args:
            - crop_size (tuple): The size of the crop (width, height) in
                pixels.
            - seed (int): A seed for the random number generator for
                reproducible results. Default is 42.
        """
        super().__init__(locals())

    @StepBase._tensor_pyfunc_wrapper
    def process_step(self, image_tensor):
        image_shape = tf.shape(image_tensor)
        crop_height, crop_width = self.parameters["crop_size"]

        crop_height = tf.minimum(crop_height, image_shape[0])
        crop_width = tf.minimum(crop_width, image_shape[1])

        random_top = tf.random.uniform(
            shape=(),
            maxval=image_shape[0] - crop_height + 1,
            dtype=tf.int32,
            seed=self.parameters["seed"],
        )
        random_left = tf.random.uniform(
            shape=(),
            maxval=image_shape[1] - crop_width + 1,
            dtype=tf.int32,
            seed=self.parameters["seed"],
        )

        cropped_image = tf.image.crop_to_bounding_box(
            image_tensor, random_top, random_left, crop_height, crop_width
        )
        return cropped_image


if __name__ == "__main__":
    step = RandomCropper()
    print(step.get_step_json_representation())
