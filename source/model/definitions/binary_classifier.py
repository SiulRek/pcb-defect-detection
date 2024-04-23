from enum import Enum

from source.model.definitions.image_classifier import FIGURES, METRICS, STATISTICS


class CATEGORY_CODES(Enum):
    NO_DEFECT = 0
    DEFECT = 1
