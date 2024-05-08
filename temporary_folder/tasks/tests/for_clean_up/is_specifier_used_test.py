import unittest

from temporary_folder.tasks.helpers.for_cleanup.is_specifier_used import is_specifier_used


class TestIsSpecifierUsed(unittest.TestCase):
    def test_function_call(self):
        code_text = "example(name)"
        self.assertTrue(is_specifier_used('name', code_text), "Failed to detect function call with the specifier.")

    def test_module_usage_not_including_imports_1(self):
        code_text = "import name"
        self.assertFalse(is_specifier_used('name', code_text), "Incorrectly detected name used as a module without including imports.")

    def test_module_usage_not_including_imports_2(self):
        code_text = "import module as name"
        self.assertFalse(is_specifier_used('name', code_text), "Incorrectly detected name used as a module without including imports.")

    def test_module_usage_not_including_imports_3(self):
        code_text = "from module import name"
        self.assertFalse(is_specifier_used('name', code_text), "Incorrectly detected name used as a module without including imports.")

    def test_module_usage_including_imports_1(self):
        code_text = "import name"
        self.assertTrue(is_specifier_used('name', code_text, include_imports=True), "Failed to detect name used as a module when including imports.")
    
    def test_module_usage_including_imports_2(self):
        code_text = "import module as name"
        self.assertTrue(is_specifier_used('name', code_text, include_imports=True), "Failed to detect name used as a module when including imports.")
    
    def test_module_usage_including_imports_3(self):
        code_text = "from module import name"
        self.assertTrue(is_specifier_used('name', code_text, include_imports=True), "Failed to detect name used as a module when including imports.")

    def test_passed_as_argument(self):
        code_text = "other_function(name)"
        self.assertTrue(is_specifier_used('name', code_text), "Failed to detect name passed as an argument.")

    def test_assignment(self):
        code_text = "name = 'value'"
        self.assertTrue(is_specifier_used('name', code_text), "Failed to detect name in assignment.")

    def test_not_used(self):
        code_text = "sample = 'value'"
        self.assertFalse(is_specifier_used('name', code_text), "Incorrectly detected name when not used.")

    def test_in_comment(self):
        code_text = "# This is a comment with name"
        self.assertFalse(is_specifier_used('name', code_text), "Incorrectly detected name used in a comment.")

    def test_in_string_literal(self):
        code_text = "print('name is a common word')"
        self.assertFalse(is_specifier_used('name', code_text), "Incorrectly detected name used in a string literal.")

    def test_with_similar_names(self):
        code_text = "username = 'value'"
        self.assertFalse(is_specifier_used('name', code_text), "Incorrectly detected name when similar names are present.")

    def test_multiple_occurrences(self):
        code_text = """
name = 'value'
def function():
    return name()
name.attribute
"""
        self.assertTrue(is_specifier_used('name', code_text), "Failed to detect multiple valid usages of name.")

if __name__ == '__main__':
    unittest.main()
