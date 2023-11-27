
import source.image_preprocessing.preprocessing_steps as steps 

STEP_CLASS_MAPPING = {
    'Adaptive Histogram Equalizer': steps.AdaptiveHistogramEqualizer,
    'Global Histogram Equalizer': steps.GlobalHistogramEqualizer,
    'Gaussian Blur Filter': steps.GaussianBlurFilter,
    'Median Blur Filter': steps.MedianBlurFilter,
    'Bilateral Filter': steps.BilateralFilter,
    'Average Blur Filter': steps.AverageBlurFilter,
    'Non Local Mean Denoiser': steps.NLMeanDenoiser,
    'RGB To Grayscale': steps.RGBToGrayscale,
    'Grayscale To RGB': steps.GrayscaleToRGB,
    'Otsu Thresholding': steps.OstuThresholder,
    'Adaptive Thresholding': steps.AdaptiveThresholder,
    'Binary Thresholding': steps.BinaryThresholder,
    'Truncated Thresholding': steps.TruncatedThresholder,
    'Threshold to Zero': steps.ZeroThreshold,
    'Erosion Filter': steps.ErosionFilter,
    'Dilation Filter': steps.DilationFilter,
    'Dilate Erode Sequencer': steps.DilateErodeSequencer,
    'Square Shape Padder': steps.SquareShapePadder,
    'Shape Resizer': steps.ShapeResizer,
    'Min Max Normalizer': steps.MinMaxNormalizer,
    'Standard Normalizer': steps.StandardNormalizer,
    'Mean Normalizer': steps.MeanNormalizer
}