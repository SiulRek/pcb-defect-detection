import os


from temporary_folder.tasks.helpers.general.extractor_base import ExtractorBase

# from temporary_folder.tasks.helpers.for_cleanup.line_validation import (
# )

#----- Definitions to be extracted later -------------
SELECT_ONLY_TAG = "#only"
SELECT_NOT_TAG = "#not"
CHECKPOINTS_TAG  = "#checkpoints"

def line_validation_for_select_only(line):
    if SELECT_ONLY_TAG in line:
        options = line.replace(SELECT_ONLY_TAG, "").strip().split(",")
        options = [option.strip().upper() for option in options]
        return options


def line_validation_for_select_not(line):
    if SELECT_NOT_TAG in line:
        options = line.replace(SELECT_NOT_TAG, "").strip().split(",")
        options = [option.strip().upper() for option in options]
        return options


def line_validation_for_checkpoints(line):
    return CHECKPOINTS_TAG in line

from enum import Enum

class REFERENCE_TYPE(Enum):
    SELECT_ONLY = "select_only"
    SELECT_NOT = "select_not"
    CHECKPOINTS = "checkpoints"

# ---------------------------------------------------

class ReferencedContentExtractor(ExtractorBase):
    
    def handler_select_only(self, line):
        if result := line_validation_for_select_only(line):
            return (REFERENCE_TYPE.SELECT_ONLY, result)
        return None
    
    def handler_select_not(self, line):
        if result := line_validation_for_select_not(line):
            return (REFERENCE_TYPE.SELECT_NOT, result)
        return None

    def handler_checkpoints(self, line):
        if line_validation_for_checkpoints(line):
            return (REFERENCE_TYPE.CHECKPOINTS, None)
        return None
    
    def post_process_referenced_contents(self, referenced_contents):
        select_not = []
        select_only = []
        checkpoint_tag = False
        for ref_type, content in referenced_contents:
            if ref_type == REFERENCE_TYPE.SELECT_NOT:
                select_not.extend(content)
            elif ref_type == REFERENCE_TYPE.SELECT_ONLY:
                select_only.extend(content)
            elif ref_type == REFERENCE_TYPE.CHECKPOINTS:
                checkpoint_tag = True

        select_not = list(set(select_not)) or None
        select_only = list(set(select_only)) or None
        return select_only, select_not, checkpoint_tag
            

