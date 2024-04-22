from source.utils.pcb_visualization import PCBVisualizerforCV2, PCBVisualizerforTF
from source.utils.get_sample_from_distribution import get_sample_from_distribution
from source.utils.logger import Logger
from source.utils.test_result_logger import TestResultLogger

# This is a Windows-specific import
from os import name
if name == 'nt':
    from source.utils.simple_popup_handler import SimplePopupHandler
else:
    from unittest.mock import MagicMock
    SimplePopupHandler = MagicMock