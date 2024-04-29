import os


def get_response_file_path(root_dir):
    return os.path.join(root_dir, "local", "scripts", "generate", "answer.txt")


def get_temporary_file_path(root_dir):
    return os.path.join(root_dir, "local", "scripts", "generate", "temporary_file.txt")


def get_fill_text_directory(root_dir):
    return os.path.join(root_dir, "temporary_folder", "tasks", "fill_texts")


def get_python_environment_path(root_dir):
    return os.path.join(root_dir, "venv")