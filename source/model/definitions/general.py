from enum import Enum

STRING_SEPARATOR = "-"

RESULT_FILE_NAME = "evaluation_results.pkl"


class REPORT_ELEMENT(Enum):
    HEADER = "header"
    TITLE = "title"
    SUBTITLE = "subtitle"
    TEXT = "text"
    RED_TEXT = "red_text"
    FIGURE = "figure"
    TABLE = "table"
    LINK = "link"
