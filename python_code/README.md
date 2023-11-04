# python_code Directory

This directory contains the core modules and scripts that make up the PCB Defect Detection system. Each subdirectory and file has a specific role in the workflow, from preprocessing images to model training and hyperparameter tuning.

## Contents

- [hyperparameter_tuning](./hyperparameter_tuning): Contains scripts and modules used for optimizing preprocessing and model parameters to enhance performance.
- [image_preprocessing](./image_preprocessing): Holds the pipeline and necessary steps for preparing images for model input, including noise reduction, normalization, and thresholding.
- [load_raw_data](./load_raw_data): Scripts that handle loading and preparation of raw PCB image data.
- [model](./model): Includes the implementation of training scripts of the PCB defect detection model.
- [utils](./utils): Utility functions and helper scripts that are general-purpose modules that can be used throughout the project for tasks like image visualization or logging.
- [main.py](./main.py): The main executable script that integrates all modules for training, evaluation, or prediction.
