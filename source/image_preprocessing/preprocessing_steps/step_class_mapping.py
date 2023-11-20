
from source.image_preprocessing.preprocessing_steps import AdaptiveHistogramEqualizer, GlobalHistogramEqualizer, GaussianBlurFilter 

STEP_CLASS_MAPPING = {
    'Adaptive Histogram Equalization': AdaptiveHistogramEqualizer,
    'Global Histogram Equalization': GlobalHistogramEqualizer,
    'Gaussian Blurring': GaussianBlurFilter
}