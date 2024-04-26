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
    
    #S What is the output of the following code?
    # reference_1.py
    #C Wow Thats a good comment
    # reference_2.py
    #C This is another comment
    # preprocessing/tests/test_runner.py
    # reference_3.txt
    #Err
    #E The end of the code