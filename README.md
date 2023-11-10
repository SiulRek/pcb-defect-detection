# PCB Defect Detection Project

## Overview
This project is part of my System Test Engineering coursework, supervised by Ms. Schappacher. The primary objectives of this project are to:

- Define and understand different types of PCB defects.
- Develop a comprehensive data preparation strategy for PCB defect detection.
- Construct a suitable machine-learning model in Python for defect identification.
- Implement model training and evaluation mechanisms to ensure model accuracy.
- Validate the model's applicability and reliability on a new dataset.

## Project Structure
The repository is organized as follows:

- **[python_code](./python_code):** Contains Python scripts and Jupyter notebooks for data analysis, model development, and image preprocessing.

- **[data](./data):** Contains the dataset used for training and testing and processed data.

- **[documentation](./documentation):** Contains project documentation, reports, and any research materials.

- **[environment](./environment):** Contains information to set up the project environment.

- **[outputs](./outputs):** Contains preprocessing and training outputs.

- **[references](./references):** Contains external references.

- **[results](./results):** Stores model evaluation results.

## Getting Started
To get started with the PCB Defect Detection project, follow these instructions:

1. **Clone the Repository**  
   `git clone https://github.com/SiulRek/pcb-defect-detection.git`  
   Navigate into the project directory:  
   `cd pcb-defect-detection`

2. **Set Up Python Virtual Environment**  
   Create the virtual environment:  
   `python -m venv venv`  
   Activate the virtual environment:  
   On Windows:  
   `.\venv\Scripts\activate`  
   On macOS and Linux:  
   `source venv/bin/activate`

3. **Configure Repository Path**  
   To include the repository path in the system's search paths at runtime, create a `your-repository-name.pth` file containing the absolute path to the `your-repository-name` directory. Place this file in the `Lib\site-packages` directory of your Python installation or virtual environment. This ensures that Python includes your repository directory in its `sys.path`.
   
   Example of `.pth` file content:  
   `C:\path\to\your\repository`


4. **Install Dependencies**  
   Install the required packages:  
   `pip install -r environment/requirements.txt`

5. **Generate TFRecord Files**  
   Run the script to create the Tensorflow Record files of the repository datasets:  
   [Create TFRecords Script](./python_code/load_raw_data/create_tf_records_local.py)


