from enum import Enum

class FIGURES(Enum):
    HISTORY = 'history'
    CONFUSION_MATRIX = 'confusion_matrix'
    ROC_CURVE = 'roc_curve'
    ALL_RESULTS = 'all_results'
    FALSE_RESULTS = 'false_results'
    SPECIFIC_RESULTS = 'specific_results'
    EVALUATION_METRICS = 'evaluation_metrics'
    MODEL_CONFIGURATION = 'model_configuration'

class METRICS(Enum):
    ACCURACY = 'accuracy'
    PRECISION = 'precision'
    RECALL = 'recall'
    F1 = 'f1'

class STATISTICS(Enum):
    MEAN = 'mean'
    STD = 'std'
    MIN = 'min'
    MAX = 'max'
    MEDIAN = 'median'