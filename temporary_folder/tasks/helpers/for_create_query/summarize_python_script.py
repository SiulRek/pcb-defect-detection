import ast


def summarize_python_file(file_path, include_definitions_without_docstrings=False):
    """
    Summarizes a Python file by extracting classes, functions, and their
    respective docstrings, maintaining proper indentation to reflect the
    structure.

    Args:
        - file_path (str): The path to the Python file to be summarized.
        - include_definitions_without_docstrings (bool, optional): Whether
            to include classes andfunctions without docstrings in the summary.
            Defaults to False.

    Returns:
        - str: A summary of the classes and functions with docstrings.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        tree = ast.parse(file.read(), filename=file_path)

    def get_docstring(node):
        """Utility function to get the docstring of a node if it exists."""
        return ast.get_docstring(node) or ""

    def format_docstring(docstring, indent):
        """Format the docstring to include it in the summary properly indented."""
        if docstring:
            indent_space = " " * indent
            formatted = "\n".join(
                [f'{indent_space}    """']
                + [f"{indent_space}    {line}" for line in docstring.split("\n")]
                + [f'{indent_space}    """']
            )
            return formatted
        return ""

    def summarize_node(node, indent=0):
        """Recursively summarize a node."""
        summary = []
        if isinstance(node, ast.ClassDef):
            docstring = get_docstring(node)
            if not docstring and not include_definitions_without_docstrings:
                return []
            summary.append(f"{' ' * indent}class {node.name}:")
            summary.append(format_docstring(docstring, indent))
            for child in node.body:
                summary.extend(summarize_node(child, indent + 4))

        elif isinstance(node, ast.FunctionDef):
            docstring = get_docstring(node)
            if not docstring and not include_definitions_without_docstrings:
                return []
            args = ", ".join(arg.arg for arg in node.args.args)
            summary.append(f"{' ' * indent}def {node.name}({args}):")
            summary.append(format_docstring(docstring, indent))

        return summary

    summary = []
    for node in tree.body:
        summary.extend(summarize_node(node))

    return "\n".join(summary)


if __name__ == "__main__":
    path = r"path/to/summarize_python_script.py"
    print(summarize_python_file(path))
