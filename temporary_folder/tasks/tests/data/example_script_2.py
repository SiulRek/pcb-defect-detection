class FileManager:
    """Class to handle file management operations."""
    
    def __init__(self, directory):
        self.directory = directory
    
    def create_file(self, filename, content):
        """Create a file with the given name and content in the directory."""
        with open(f"{self.directory}/{filename}", 'w') as file:
            file.write(content)
        print("File created successfully.")
    
    def delete_file(self, filename):
        """Delete a file with the given name from the directory."""
        import os
        os.remove(f"{self.directory}/{filename}")
        print("File deleted successfully.")

class Calculator:
    """Simple calculator class to perform basic arithmetic operations."""
    
    def __init__(self):
        pass

    @staticmethod
    def add(a, b):
        """Return the sum of two numbers."""
        return a + b
    
    @staticmethod
    def subtract(a, b):
        """Return the difference of two numbers."""
        return a - b

def main():
    """Main function to execute some operations."""
    fm = FileManager('/tmp')
    fm.create_file('test.txt', 'Hello, World!')
    fm.delete_file('test.txt')
    
    calc = Calculator()
    print("Sum:", Calculator.add(5, 3))
    print("Difference:", Calculator.subtract(5, 3))

if __name__ == "__main__":
    main()
