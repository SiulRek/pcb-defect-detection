# Image Preprocessing for PCB Analysis

This folder is dedicated to the preprocessing of PCB (Printed Circuit Board) images as part of the PCB defect detection project. It includes various components such as preprocessing steps, pipelines and test suites.

## Overview

The `image_preprocessing` folder consists of the following subdirectories and files:

- [notebooks](./notebooks): Jupyter notebooks for image preprocessing experimenting, evaluation and documentation.
- [pipelines](./pipelines): Pipelines with exact hyperparameters and hyperparameter ranges to be loaded with the `ImagePreprocessor`.
- [preprocessing_steps](./preprocessing_steps): Modules defining the preprocessing steps to be applied to images and to be integrated within the image preprocessing framwork.
- [tests](./tests): Test cases for verification of the functionality of the preprocessing steps.
- [image_preprocessor.py](./image_preprocessor.py): Contains the `ImagePreprocessor` class. The main handler that defines the image preprocessing operations and is able to integrate preprocessing steps to the pipeline.


## Usage
```python
# Initialize the image preprocessor
preprocessor = ImagePreprocessor()

# Make pipeline
pipeline = [
    StepOne(**params_one),
    StepTwo(**params_two)
]

# Set pipeline
preprocessor.set_pipe(pipeline)
 
# Process a dataset
processed_dataset = preprocessor.process(image_dataset)
```

## Interactive Demonstration
Please refer to the interactive Jupyter notebook [image_preprocessor_documentation.ipynb](./notebooks/image_preprocessor_documentation.ipynb) that provides an in-depth explanation and practical application insights into the image preprocessing framework.



