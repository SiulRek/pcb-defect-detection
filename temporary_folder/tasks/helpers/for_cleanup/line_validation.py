from temporary_folder.tasks.constants.definitions import CLEANUP_TAGS as TAGS

SELECT_ONLY_TAG = TAGS.SELECT_ONLY.value
SELECT_NOT_TAG = TAGS.SELECT_NOT.value
CHECKPOINTS_TAG = TAGS.CHECKPOINTS.value


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
