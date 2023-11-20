
import source.image_preprocessing.preprocessing_steps as steps 

STEP_CLASS_MAPPING = {
    'Adaptive Histogram Equalization': steps.AdaptiveHistogramEqualizer,
    'Global Histogram Equalization': steps.GlobalHistogramEqualizer,
    'Gaussian Blurring': steps.GaussianBlurFilter,
    'Median Blurring': steps.MedianBlurFilter
}