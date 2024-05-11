def separate_imports(code_text):
    """
    Reads Python code from a string and separates the lines into import
    statements and other code.

    This function handles both single-line and multi-line import statements.
    Multi-line imports are assumed to start with 'from ... import (' or 'import
    (' and end with ')'. Lines that are part of a multi-line import statement
    are grouped together in the import list. All other lines, including blank
    lines outside of import statements, are categorized separately.

    Args:
        - code_text (str): A string containing Python code.

    Returns:
        - tuple: A tuple containing two lists:
            - First list contains lines that are import statements.
            - Second list contains all other lines.
    """
    import_statements = []
    other_code = ""
    previous_line_is_import = False

    lines = iter(code_text.splitlines(keepends=True))

    try:
        while True:
            line = next(lines)
            stripped_line = line.strip()
            if stripped_line.startswith("import ") or stripped_line.startswith("from "):
                import_statement = line
                previous_line_is_import = True
                if "(" in line and ")" not in line:
                    continue_reading_import = True
                    while continue_reading_import:
                        try:
                            line = next(lines)
                            import_statement += line
                            if ")" in line:
                                continue_reading_import = False
                        except StopIteration:
                            continue_reading_import = False
                import_statements.append(import_statement.strip())
            else:
                if not previous_line_is_import or not line.strip() == "":
                    other_code += line
                if line.strip():
                    previous_line_is_import = False
    except StopIteration:
        pass

    return import_statements, other_code


if __name__ == "__main__":
    file_path = r"temporary_folder/tasks/tests/data/example_script_3.py"
    with open(file_path, "r") as file:
        code_text = file.read()
    import_statements, other_code = separate_imports(code_text)
    print("Import statements:")
    print("\n".join(import_statements))
    print("\nOther code:")
    print(other_code)
