import os
import sys

from temporary_folder.tasks.constants.getters import (
    get_checkpoint_directory,
    get_environment_path,
)
from temporary_folder.tasks.helpers.for_cleanup.cleanup_file import cleanup_file
from temporary_folder.tasks.helpers.for_cleanup.referenced_contents_extractor import (
    ReferencedContentExtractor,
)
import temporary_folder.tasks.helpers.general.print_statements as task_prints

if len(sys.argv) == 3:
    ROOT_DIR = sys.argv[1]
    FILE_PATH = sys.argv[2]
    sys.path.append(ROOT_DIR)
else:
    ROOT_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."
    )
    FILE_PATH = os.path.join(
        ROOT_DIR,
        "temporary_folder",
        "tasks",
        "tests",
        "cleanup_test.py",
    )

extract_referenced_contents = ReferencedContentExtractor().extract_referenced_contents


def clean_up(file_path, root_dir):
    """
    Create a query from the file and referenced contents in the file.

    Args:
        - file_path (str): The path to the file to be processed.
        - root_dir (str): The root directory of the project.
    """
    checkpoint_dir = get_checkpoint_directory(root_dir)
    environment_path = get_environment_path(root_dir)

    referenced_contents, updated_content = extract_referenced_contents(
        file_path, root_dir
    )
    select_only, select_not, checkpointing = referenced_contents

    if select_only != None and select_not != None:
        msg = "Cannot have both select_only and select_not options specified."
        raise ValueError(msg)

    with open(file_path, "w") as file:
        file.write(updated_content)

    cleanup_file(
        file_path=file_path,
        select_only=select_only,
        select_not=select_not,
        checkpointing=checkpointing,
        python_env_path=environment_path,
        checkpoint_dir=checkpoint_dir,
    )


def main():
    task_prints.process_start("Cleanup")
    clean_up(FILE_PATH, ROOT_DIR)
    task_prints.process_end()


if __name__ == "__main__":
    main()
