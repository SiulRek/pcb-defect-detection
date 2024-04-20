from source.utils.pcb_visualization import PCBVisualizerforCV2, PCBVisualizerforTF
from source.utils.recursive_type_conversion import recursive_type_conversion
from source.utils.class_instances_serializer import ClassInstancesSerializer
from source.utils.parse_and_repeat import parse_and_repeat
from source.utils.get_sample_from_distribution import get_sample_from_distribution
from source.utils.logger import Logger
from source.utils.test_result_logger import TestResultLogger
from source.utils.copy_json_exclude_entries import copy_json_exclude_entries

# This is a Windows-specific import
from os import name
if name == 'nt':
    from source.utils.simple_popup_handler import SimplePopupHandler
else:
    from unittest.mock import MagicMock
    SimplePopupHandler = MagicMock