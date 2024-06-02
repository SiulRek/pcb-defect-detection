import tensorflow as tf

from source.preprocessing.helpers.for_steps.step_base import StepBase


class GaussianNoiseInjector(StepBase):
    """
    A data augmentation step that injects Gaussian noise into an image tensor.
    The noise intensity can be specified. The output tensor type is tf.float32.
    An optional boolean argument 'apply_clipping' controls whether to clip the
    output values.
    """

    arguments_datatype = {
        "mean": float,
        "sigma": float,
        "apply_clipping": bool,
        "seed": int,
    }
    name = "Gaussian Noise Injector"

    def __init__(self, mean=0.0, sigma=0.05, apply_clipping=True, seed=42):
        """
        Initializes the GaussianNoiseInjector object for integration into an
        image preprocessing pipeline.

        Args:
            - mean (float): The mean of the Gaussian noise distribution.
                Default is 0.0.
            - sigma (float): The standard deviation of the Gaussian noise
                distribution. Default is 0.05.
            - apply_clipping (bool): If True, clips the output values to be
                within a valid range. Default is True.
            - seed (int): Random seed for reproducibility. Default is 42.
        """
        super().__init__(locals())

    @StepBase._tensor_pyfunc_wrapper
    def process_step(self, image_tensor):
        shape = tf.shape(image_tensor)
        gaussian_noise = tf.random.normal(
            shape,
            mean=self.parameters["mean"],
            stddev=self.parameters["sigma"],
            seed=self.parameters["seed"],
        )
        gaussian_noise = tf.cast(gaussian_noise, self.output_datatype)
        noisy_image = image_tensor + gaussian_noise

        if self.parameters["apply_clipping"]:
            if self.output_datatype == tf.uint8:
                noisy_image = tf.clip_by_value(noisy_image, 0, 255)
            else:
                noisy_image = tf.clip_by_value(noisy_image, 0.0, 1.0)

        return noisy_image


if __name__ == "__main__":
    step = GaussianNoiseInjector()
    print(step.get_step_json_representation())
