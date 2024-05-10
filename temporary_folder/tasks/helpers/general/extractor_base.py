class ExtractorBase:
    """
    Base class for extracting referenced content from files based on specified handlers.

    This class scans files for specific content patterns using a series of handler methods
    prefixed with 'handler_'. Each handler processes a line from the file to detect and extract
    different types of referenced content. The extracted content can include references to other
    files, scripts, errors, or any specific tags defined within the handler methods.

    Methods:
        extract_referenced_contents: Extract referenced contents from a file and separates them from non-referenced content.
        post_process_referenced_contents: Provide a hook for child classes to further process the referenced contents.
    """
    def __init__(self):
        self.initialize_handlers()
    
    def initialize_handlers(self):
        """
        Initializes the handlers for extracting referenced content.
        """
        self.handlers  = [
                getattr(self, method)
                for method in dir(self)
                if callable(getattr(self, method)) and method.startswith("handler_")
            ]
    
    def _extract_referenced_contents(self, text):
        """
        Extracts referenced and updated text from the input text.

        Args:
            text (str): The text to extract referenced content from.
        
        Returns:
            tuple: A tuple containing a list of referenced contents and the updated text.
        """
        referenced_contents = []
        updated_text_lines = []
        
        for line in text.splitlines():
            result = None
            stripped_line = line.strip()
            for handler in self.handlers:
                if result := handler(stripped_line):
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
        Extracts referenced contents from a specified file, maintaining the order of their occurrence.

        This method iterates through each line of the file, applying handler methods that start
        with 'handler_' to extract specific content based on tags or patterns. Each handler can
        return a single item or a list of items, which are collected into a list of referenced contents.
        Non-referenced lines are gathered into a separate list.

        Args:
            file_path (str): The path to the file from which to extract content.
            root_dir (str): The root directory path, used to resolve relative paths and environment settings.

        Returns:
            tuple: A tuple containing a list of referenced contents and the updated text.
        """
        self.file_path = file_path
        self.root_dir = root_dir

        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            referenced_contents, updated_text = self._extract_referenced_contents(text)

        referenced_contents = self.post_process_referenced_contents(referenced_contents)
        return referenced_contents, updated_text

    def post_process_referenced_contents(self, referenced_contents):
        """
        Processes the collected referenced contents for final output.

        This method can be overridden by child classes to implement custom processing of the
        referenced contents, such as merging related items or filtering out specific results.
        By default, it returns the referenced contents as they were collected.

        Args:
            referenced_contents (list): The list of referenced contents collected by the handlers.

        Returns:
            list: The processed list of referenced contents.
        """
        return referenced_contents
