from enum import Enum

class Category(Enum):
    """ Enum class for PCB defect categories."""
    NO_DEFECT = 0
    MISSING_HOLE = 1
    MOUSE_BITE = 2
    OPEN_CIRCUIT = 3
    SHORT = 4
    SPUR = 5
    SPURIOUS_COPPER = 6
    PIN_HOLE = 7