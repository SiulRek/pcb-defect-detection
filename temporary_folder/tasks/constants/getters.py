import os

def get_temporary_file_path(root_dir):
    return os.path.join(root_dir, "local", "scripts", "generate", "temporary_file.txt")