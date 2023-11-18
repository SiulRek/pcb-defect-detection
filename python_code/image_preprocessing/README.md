# Image Preprocessing for PCB Analysis

This folder is dedicated to the preprocessing of PCB (Printed Circuit Board) images as part of a larger machine learning pipeline. It includes various components such as preprocessing steps, pipelines and test suites.

## Structure

The `image_preprocessing` folder consists of the following subdirectories and files:

- [pipelines](./pipelines): Pipelines with exact Hyperparameters and Hyperparameter ranges of the image preprocessor.
- [notebooks](./notebooks): Jupyter notebooks for image preprocessing evaluation and experimenting.
- [preprocessing_steps](./preprocessing_steps): Modules defining the preprocessing steps to be applied to images.
- [tests](./tests): Test cases for verification of the functionality of the preprocessing steps.
- [image_preprocessor.py](./image_preprocessor.py): The main script that defines the image preprocessing pipeline and integrates the preprocessing steps.


## Usage
```python
# Initialize the image preprocessor
preprocessor = ImagePreprocessor()

# Make pipeline
pipeline = [
    StepOne(params_one),
    StepTwo(params_two)
]

# Set pipeline
preprocessor.set_pipe(pipeline)

# Process a dataset
processed_dataset = preprocessor.process(image_dataset)
