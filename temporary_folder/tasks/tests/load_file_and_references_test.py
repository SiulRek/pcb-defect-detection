"""
TODO:
    1. Run the task Load File and References
    2. check the temporary file for the output.
"""

class Car:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model

    def __eq__(self, other):
        return self.brand == other.brand and self.model == other.model

    def __str__(self):
        return f"{self.brand} {self.model}"

    def get_car(self):
        return self.brand, self.model

    #S Hello Darkness my Old Friend
    # reference_1.py
    #*test
    #summarize data/example_script_2.py
    #T Comment Title
    #C Wow Thats a good comment
    #C The line continues even here
    #C The comment is so long.
    #run data/example_script.py
    #T Reference File Title
    # reference_2.py, data/reference_3.txt
    #File
    #tree . (2, False, [temp; log])
    #pylint data/example_script.py
    #C This is another comment
    # preprocessing/tests/test_runner.py
    #L
    #E The end of the code
    #makequery (100)
    #checksum 20