
import source.image_preprocessing.preprocessing_steps as steps 

STEP_CLASS_MAPPING = {
    'Adaptive Histogram Equalizer': steps.AdaptiveHistogramEqualizer,
    'Global Histogram Equalizer': steps.GlobalHistogramEqualizer,
    'Gaussian Blur Filter': steps.GaussianBlurFilter,
    'Median Blur Filter': steps.MedianBlurFilter,
    'Bilateral Filter': steps.BilateralFilter,
    'Average Blur Filter': steps.AverageBlurFilter,
}