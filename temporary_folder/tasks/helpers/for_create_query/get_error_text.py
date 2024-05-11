from temporary_folder.tasks.constants.definitions import TEST_RESULTS_FILE
from temporary_folder.tasks.constants.patterns import CREATE_QUERY_PATTERNS
from temporary_folder.tasks.helpers.general.find_file import find_file

TEST_RESULT_PATTERN = CREATE_QUERY_PATTERNS.TEST_RESULT.value


def _extract_error_messages(log_text):

    lines = log_text.split("\n")
    messages = [""]
    for line in lines:
        if TEST_RESULT_PATTERN.match(line):
            messages.append(line)
        else:
            messages[-1] += "\n" + line

    captured_messages = []
    for message in messages:
        if not "Test passed" in message:
            captured_messages.append(message)
    if not captured_messages:
        msg = "No error messages found in log text."
        raise ValueError(msg)
    return "\n".join(captured_messages)


def get_error_text(root_dir, reference_file_path):
    """
    Function to extract the error text from the log file.

    Args:
        - root_dir (str): The root directory of the project.
        - reference_file_path (str): The path to the file.

    Returns:
        - str: The error text.
    """
    log_path = find_file(TEST_RESULTS_FILE, root_dir, reference_file_path)
    with open(log_path) as f:
        log_text = f.read()
    return _extract_error_messages(log_text)
