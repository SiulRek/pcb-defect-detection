import os

from temporary_folder.tasks.helpers.for_create_query.line_validation import (
    line_validation_for_checksum,
)
from temporary_folder.tasks.helpers.general.extract_python_code import (
    extract_python_code,
)
from temporary_folder.tasks.helpers.general.make_query import make_query


def write_to_file(file_path, content):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)


class Finalizer:
    """
    Processes final stages of file content modification, including checksum
    verification, query creation, and response handling.

    Attributes:
        - file_path (str): Path to the target file for final content.
        - query_path (str): Path to save the generated query.
        - response_path (str): Path to save the query response.
        - checksum (int): Expected checksum for validation.
        - updated_contents (str): Content to be processed and validated.
        - final_lines (list): Lines of content after processing.
        - query (str): Query to be executed.
    """

    def __init__(self):
        """Initializes the Finalizer with default paths and content states."""
        self.file_path = None
        self.query_path = None
        self.response_path = None
        self.checksum = None
        self.updated_contents = None
        self.final_lines = []
        self.query = None

    def set_paths(self, file_path, query_path, response_path):
        """
        Sets the file, query, and response paths.

        Args:
            - - file_path (str): Target file path.
            - - query_path (str): Output query file path.
            - - response_path (str): Query response file path.
        """
        self.file_path = file_path
        self.query_path = query_path
        self.response_path = response_path

    def _validate_lines(self):
        """Validates each line of the updated contents for checksums and
        updatesfinal lines."""
        for line in self.updated_contents.splitlines():
            if result := line_validation_for_checksum(line.strip()):
                self.checksum = result
            else:
                self.final_lines.append(line)

    def _verify_checksum(self):
        """Verifies if the line difference matches the expected checksum, raisingan
        error on mismatch."""
        if self.checksum:
            with open(self.file_path, "r", encoding="utf-8") as file:
                file_contents = file.read().splitlines()
                diff = len(file_contents) - len(self.final_lines)
                if diff != self.checksum:
                    msg = f"Checksum mismatch: {diff} != {self.checksum}"
                    raise ValueError(msg)

    def _handle_query(self, make_query_kwargs):
        """
        Generates a query based on provided parameters and handles the execution
        and file output of the response.

        Args:
            - make_query_kwargs (dict): Parameters including
                'create_python_script' and 'max_tokens' for the query.
        """
        create_python_script = make_query_kwargs["create_python_script"]
        max_tokens = make_query_kwargs["max_tokens"]
        print("Making query...")
        response = (
            make_query(self.query, max_tokens) if max_tokens else make_query(self.query)
        )
        write_to_file(self.response_path, response)
        print(f"Response saved to {self.response_path}")
        if create_python_script:
            code = extract_python_code(response)
            python_path = os.path.splitext(self.response_path)[0] + ".py"
            write_to_file(python_path, code)
            print(f"Python script saved to {python_path}")

    def _finalize(self, make_query_kwargs):
        """
        Writes the final processed content to the file and handles the query and
        its response.

        Args:
            - make_query_kwargs (dict): Arguments for making the query.
        """
        final_text = "\n".join(self.final_lines)
        write_to_file(self.file_path, final_text)
        write_to_file(self.query_path, self.query)
        print(f"Query saved to {self.query_path}")
        if make_query_kwargs:
            self._handle_query(make_query_kwargs)

    def finalize(self, updated_contents, query, make_query_kwargs):
        """
        Finalizes the content processing by validating, verifying checksum, and
        writing results.

        Args:
            - - updated_contents (str): The updated content to process.
            - - query (str): The query to handle.
            - - make_query_kwargs (dict): Additional parameters for
                queryhandling.
        """
        self.updated_contents = updated_contents
        self.query = query
        self._validate_lines()
        self._verify_checksum()
        self._finalize(make_query_kwargs)
