
from python_code.image_preprocessing.preprocessing_steps import AdaptiveHistogramEqualization, GlobalHistogramEqualization, GaussianBlurring 

STEP_CLASS_MAPPING = {
    'Adaptive Histogram Equalization': AdaptiveHistogramEqualization,
    'Global Histogram Equalization': GlobalHistogramEqualization,
    'Gaussian Blurring': GaussianBlurring
}