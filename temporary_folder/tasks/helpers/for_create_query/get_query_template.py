import os

from temporary_folder.tasks.constants.getters import get_query_templates_directory


def get_query_template(name, root_dir):
    """
    Get a query template by name.

    Args:
        - name (str): The name of the query template.
        - root_dir (str): The root directory of the project.

    Returns:
        - str: The query template.
    """
    query_template_directory = get_query_templates_directory(root_dir)
    for file in os.listdir(query_template_directory):
        base, _ = os.path.splitext(file)
        if base == name:
            file_path = os.path.join(query_template_directory, file)
            with open(file_path, "r", encoding="utf-8") as query_template_file:
                query_template = query_template_file.read()
            return query_template
    msg = f"Query template with name {name} not found."
    raise FileNotFoundError(msg)
