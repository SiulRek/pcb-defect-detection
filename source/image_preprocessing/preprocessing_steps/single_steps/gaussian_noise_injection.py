import tensorflow as tf
from source.image_preprocessing.preprocessing_steps.step_base import StepBase


class GaussianNoiseInjection(StepBase):
    """
    A data augmentation step that injects Gaussian noise into an image tensor.
    The noise intensity can be specified, and the output tensor type is tf.float32.
    """

    arguments_datatype = {'mean': float, 'sigma': float}
    name = 'Gaussian Noise Injection'

    def __init__(self, mean=0.0, sigma=0.05):
        """
        Initializes the GaussianNoiseInjection object for integration into an image preprocessing pipeline.

        Args:
            mean (float): The mean of the Gaussian noise distribution. Default is 0.0.
            sigma (float): The standard deviation of the Gaussian noise distribution. Default is 0.05.
        """
        super().__init__(locals())

    @StepBase._tensor_pyfunc_wrapper
    def process_step(self, image_tensor):
        shape = tf.shape(image_tensor)
        gaussian_noise = tf.random.normal(shape, mean=self.parameters['mean'], stddev=self.parameters['sigma'])
        gaussian_noise = tf.cast(gaussian_noise, self.output_datatype)
        noisy_image = image_tensor + gaussian_noise
        if self.output_datatype == tf.uint8:
            noisy_image = tf.clip_by_value(noisy_image, 0, 255)
        else:
            noisy_image = tf.clip_by_value(noisy_image, 0.0, 1.0)
        return noisy_image

if __name__ == '__main__':
    step = GaussianNoiseInjection()
    print(step.get_step_json_representation())
