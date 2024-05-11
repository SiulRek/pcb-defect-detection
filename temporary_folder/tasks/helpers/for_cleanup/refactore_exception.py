import re

from temporary_folder.tasks.constants.definitions import LINE_WIDTH
from temporary_folder.tasks.helpers.general.wrap_text import wrap_text


def refactor_exception(code):
    """
    Refactors the exception code in the provided code. The refactored code will
    have the exception message stored in a variable before raising the
    exception.

    Args:
        - code (str): The code to be refactored.

    Returns:
        - str: The refactored code.
    """
    lines = code.splitlines()
    updated_lines = []
    for i, line in enumerate(lines):
        match = re.search(
            r"(\s*)raise (\w+(Exception|Error|Warning))\((f?'.*'|f?\".*\")\)", line
        )
        if match:
            indent = match.group(1)
            exception_type = match.group(2)
            msg = match.group(4)
            if msg.startswith("f"):
                q_start = msg[:2]
                q_end = msg[1]
            else:
                q_start = msg[0]
                q_end = msg[0]
            msg = wrap_text(msg, LINE_WIDTH - len(indent) - 6)
            if len(msg.splitlines()) > 1:
                for i, msg_line in enumerate(msg.splitlines()):
                    if i == 0:
                        updated_line = f"{indent}msg = {msg_line}{q_end}"
                        updated_lines.append(updated_line)
                    elif i == len(msg.splitlines()) - 1:
                        updated_line = f"{indent}msg += {q_start}{msg_line}"
                        updated_lines.append(updated_line)
                    else:
                        updated_line = f"{indent}msg += {q_start}{msg_line}{q_end}"
                        updated_lines.append(updated_line)
            else:
                updated_line = f"{indent}msg = {msg}"
                updated_lines.append(updated_line)
            updated_line = f"{indent}raise {exception_type}(msg)"
            updated_lines.append(updated_line)
        else:
            updated_lines.append(line)
    return "\n".join(updated_lines)


def refactor_exception_from_file(file_path):
    """
    Refactors the exception code in the file.

    Args:
        - file_path (str): The path to the file to be refactored.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        code = file.read()
    updated_code = refactor_exception(code)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(updated_code)


if __name__ == "__main__":
    path = r"temporary_folder/tasks/tests/data/example_script_4.py"
    refactor_exception_from_file(path)
