class ExtractorBase:
    """
    Base class for extracting referenced content from files based on specified
    validation methods.

    This class scans files for specific content patterns using a series of
    validation methods prefixed with 'validate_'. Each validation method
    processes a line from the file to detect and extract different types of
    referenced content. The extracted content can include references to other
    files, scripts, errors, or any specific tags defined within the validation
    methods.

    Methods:
        - extract_referenced_contents: Extract referenced contents from a
            file and separates them from non-referenced content.
        - post_process_referenced_contents: Provide a hook for child classes
            to further process the referenced contents.
    """

    def __init__(self):
        self.initialize_validation_methods()

    def initialize_validation_methods(self):
        """Initializes the validation methods for extracting referenced content
        from the text."""
        self.validation_methods = [
            getattr(self, method)
            for method in dir(self)
            if callable(getattr(self, method)) and method.startswith("validate_")
        ]

    def _extract_referenced_contents(self, text):
        """
        Extracts referenced and updated text from the input text based on
        validation methods.

        Args:
            - text (str): The text to extract referenced content from.

        Returns:
            - tuple: A tuple containing a list of referenced contents and
                the updated text.
        """
        referenced_contents = []
        updated_text_lines = []

        for line in text.splitlines():
            result = None
            stripped_line = line.strip()
            for val in self.validation_methods:
                if result := val(stripped_line):
                    if isinstance(result, list):
                        referenced_contents.extend(result)
                    else:
                        referenced_contents.append(result)
                    break
            if not result:
                updated_text_lines.append(line)
        updated_text = "\n".join(updated_text_lines)
        return referenced_contents, updated_text

    def extract_referenced_contents(self, file_path, root_dir):
        """
        Extracts referenced contents from a specified file using validation
        methods, while maintaining the order of their occurrence.

        This method iterates through each line of the file, applying validation
        methods that start with 'validate_' to detect and extract specific
        content based on tags or patterns. The results may include a single item
        or an aggregated list of items, based on the complexity of the
        validation logic. The non-referenced text is updated to exclude the
        extracted content, enhancing clarity and separation.

        Args:
            - file_path (str): The path to the file from which to extract
                content.
            - root_dir (str): The root directory path, used to resolve
                relative paths and environmental settings.

        Returns:
            - tuple: A tuple where the first element is the result of
                post-processing the referenced contents, defined in child
                classes, and the second element is the updated text stripped of
                referenced content.
        """
        self.file_path = file_path
        self.root_dir = root_dir

        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            referenced_contents, updated_text = self._extract_referenced_contents(text)

        process_results = self.post_process_referenced_contents(referenced_contents)
        return process_results, updated_text

    def post_process_referenced_contents(self, referenced_contents):
        """
        Processes the collected referenced contents for final output.

        This method can be overridden by child classes to implement custom
        processing of the referenced contents, such as merging related items or
        filtering out specific results. By default, it returns the referenced
        contents as they were collected.

        Args:
            - referenced_contents (list): The list of referenced contents
                collected by the validation methods.

        Returns:
            - costum_type: A custom type or list of referenced contents
                after post-processing.
        """
        return referenced_contents
