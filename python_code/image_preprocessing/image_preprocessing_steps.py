import tensorflow as tf
import cv2

class StepBase:
    """   Base class for defining preprocessing steps for images.

    Attributes:
    - Name (str)
        A name identifier for the preprocessing step.
    - params (dict)
        A dictionary containing parameters needed for the preprocessing step.

    Methods:
    - process_step(tf_image: tf.Tensor, tf_target: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        Child class must implement this method for preprocessing functionality.

    - extract_params(local_vars: dict) -> dict:
        Extracts parameters needed for the preprocessing step.

    - tf_function_decorator(func: Callable) -> Callable:
        Decorator to wrap TensorFlow functions for mapping onto a dataset.

    - py_function_decorator(func: Callable) -> Callable:
        Decorator to wrap python functions for mapping onto a dataset using tf.py_function.

    - random_params() -> None:
        Child class can implement this method to randomize parameters.

    - _reshape_color_channel(tf_image: tf.Tensor, color_channel: str = 'gray', tf_image_comparison: tf.Tensor = None) -> tf.Tensor:
        Helper method to reshape image tensor based on color channel.

    Notes:
    - The class is designed to be subclassed for specific preprocessing steps.
    - Each child class must implement the `process_step` method.
    """

    def __init__(self, name,  local_vars):
        self.name = name
        self.params = self.extract_params(local_vars)
        
    def process_step(self, tf_image, tf_target):
        # Child class must implement this method.
        pass
    
    def extract_params(self, local_vars):

        if local_vars['set_random_params']:
            return self.random_params()
        return {key: value for key, value in local_vars.items() if key != 'self' and key != 'set_random_params'}

    @staticmethod
    def tf_function_decorator(func):
        def wrapper(self, image_dataset):
            def mapped_function(img, tgt):
                return func(self, img, tgt)
            return image_dataset.map(mapped_function)
        return wrapper

    @staticmethod
    def py_function_decorator(func):
        def wrapper(self, image_dataset):
            def mapped_function(img, tgt):
                processed_img, processed_tgt = tf.py_function(
                    func=lambda image, target: func(self, image, target),  # Lambda is used to pass self.
                    inp=[img, tgt],
                    Tout=(tf.uint8, tf.int8),
                )
                return processed_img, processed_tgt
            return image_dataset.map(mapped_function)
        return wrapper

    def random_params(self): 
        # Child class must implement this method.
        pass 

    def _reshape_color_channel(self, tf_image, color_channel='gray', tf_image_comparison=None):
        if tf_image_comparison is not None:
            return tf.reshape(tf_image, [tf_image.shape[0], tf_image.shape[1], tf_image_comparison.shape[2]])
        elif color_channel == 'gray':
            return tf.reshape(tf_image, [tf_image.shape[0], tf_image.shape[1], 1])
        elif color_channel == 'rgb':
            return tf.reshape(tf_image, [tf_image.shape[0], tf_image.shape[1], 3])
        else:
            raise ValueError(f'Color channel {color_channel} is invalid.')


class AdaptiveHistogramEqualization(StepBase):

    def __init__(self, clip_limit=2.0, tile_gridsize=(8,8),  set_random_params=False):
        super().__init__('Adaptive Histogram Equalization', locals())

    @StepBase.py_function_decorator
    def process_step(self, tf_image, tf_target):

        cv_img = tf_image.numpy().astype('uint8')

        channels = cv2.split(cv_img)

        clahe = cv2.createCLAHE(clipLimit=self.params['clip_limit'], tileGridSize=self.params['tile_gridsize'])

        clahe_channels = [clahe.apply(ch) for ch in channels]

        cv2_clahe_image = cv2.merge(clahe_channels)

        tf_clahe_image = tf.convert_to_tensor(cv2_clahe_image, dtype=tf.uint8)

        tf_clahe_image = self._reshape_color_channel(tf_clahe_image, tf_image_comparison=tf_image)

        return (tf_clahe_image, tf_target)
    
    def random_params(self):
        # self.params = {}
        pass



