import re


def refactor_exception(code):
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
            formatted_msg_line = f"{indent}msg = {msg}"
            updated_lines.append(formatted_msg_line)
            updated_lines.append(f"{indent}raise {exception_type}(msg)")
        else:
            updated_lines.append(line)
    return "\n".join(updated_lines)


def refactor_exception_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        code = file.read()
    updated_code = refactor_exception(code)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(updated_code)


if __name__ == "__main__":
    path = r"temporary_folder/tasks/tests/data/example_script_4.py"
    refactor_exception_from_file(path)
