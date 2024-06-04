# Load Raw Data

The folder `load_raw_data` are used for preprocessing and preparing raw data for the PCB Detection Project. The scripts handle activities like constructing TensorFlow records, serializing datasets, and handling project-specific datasets.

## Contents

- [get_tf_dataset.py](get_tf_dataset.py): A utility script for converting a DataFrame with file locations and category codes to a TensorFlow Dataset. It has a function for loading and decoding images to ensure they are properly prepared for the TensorFlow environment.

- [dataset_serialization.py](dataset_serialization.py): This module includes utilities for serializing TensorFlow datasets containing image-label pairs into TFRecord-compatible objects and saving these objects to TFRecord files. It also has the capability to load the TFRecord files again to recreate the dataset.

- [kaggle_dataset.py](kaggle_dataset.py) [[1]](#references): This script creates a Pandas DataFrame by reading XML annotations and photos from the Kaggle PCB defect dataset. It also includes the ability to turn this DataFrame into a TensorFlow Dataset and serialize it to TFRecord file for future use.

- [deep_pcb_tangali5201.py](deep_pcb_tangali5201.py) [[2]](#references): Tangali5201 provided the deepPCB dataset for this script. It has functions for reading dataset information, constructing pandas DataFrames for training and testing, creating a training and testing TensorFlow dataset and serializing this datasets to TFRecord files for future use.

- [create_tf_records_local.py](create_tf_records_local.py): This script automates the TensorFlow record generation for the PCB defect detection datasets. It interfaces with the Kaggle and deep_pcb_tangali5201 datasets, making it easier to generate TFRecord files. 

## References

[1] A. Khatova, "PCB Defects: Dataset for Machine Learning Analysis of Printed Circuit Board Defects", Kaggle, 2021. [Online]. Available: https://www.kaggle.com/datasets/akhatova/pcb-defects/data. Accessed December 4, 2023. Additionally a related Paper: R. Ding, L. Dai, G. Li and H. Liu, "TDD-net: a tiny defect detection network for printed circuit boards," in CAAI Transactions on Intelligence Technology, vol. 4, no. 2, pp. 110-116, 6 2019, doi: 10.1049/trit.2019.0019.

[2] S. Tang, "DeepPCB: A Dataset and Benchmark for PCB Defect Detection Using Deep Learning", GitHub Repository, 2020. [Online]. Available: https://github.com/tangsanli5201/DeepPCB/tree/master. Accessed December 4, 2023.