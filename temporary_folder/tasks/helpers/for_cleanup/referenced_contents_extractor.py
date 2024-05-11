from temporary_folder.tasks.constants.definitions import (
    CLEANUP_REFERENCE_TYPES as REFERENCE_TYPE,
)
from temporary_folder.tasks.helpers.for_cleanup.line_validation import (
    line_validation_for_select_only,
    line_validation_for_select_not,
    line_validation_for_checkpoints,
)
from temporary_folder.tasks.helpers.general.extractor_base import ExtractorBase


class ReferencedContentExtractor(ExtractorBase):

    def validate_select_only_reference(self, line):
        if result := line_validation_for_select_only(line):
            return (REFERENCE_TYPE.SELECT_ONLY, result)
        return None

    def validate_select_not_reference(self, line):
        if result := line_validation_for_select_not(line):
            return (REFERENCE_TYPE.SELECT_NOT, result)
        return None

    def validate_checkpoints_reference(self, line):
        if line_validation_for_checkpoints(line):
            return (REFERENCE_TYPE.CHECKPOINTING, True)
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
            elif ref_type == REFERENCE_TYPE.CHECKPOINTING:
                checkpoint_tag = True

        select_not = list(set(select_not)) or None
        select_only = list(set(select_only)) or None
        return select_only, select_not, checkpoint_tag
