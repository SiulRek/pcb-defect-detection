# Utils
This folder contains utility modules for the PCB defect detection project. Each module provides essential functionalities that support various aspects of the project, such as visualization, file handling, and logging. Furthermore, Each module can be imported and used independently based on the specific need within the PCB defect detection project. 

## Moduls Overview

[pcb_visualization.py](./pcb_visualization.py): Contains `PCBVisualizerforCV2` and `PCBVisualizerforTF` classes for visualizing PCB images and their processing results. Supports plotting images, image comparisons, histograms, and frequency spectra for both TensorFlow and OpenCV processed images.

[recursive_type_conversion.py](./recursive_type_conversion.py): Implements a function for recursively converting data types of values to match those of a target template. This is particularly useful when storing python values in JSON files, as this format is not able to accurately represent some Python data structures, particularly tuples and nested tuples.

[class_instances_serializer.py](./class_instances_serializer.py): Manages serialization and deserialization of class instances to and from JSON format. Useful for storing and reconstructing class instances, particularly for hyperparameter tuning and experimentation when developing machine learning models.

[get_sample_from_distribution.py](./get_sample_from_distribution.py): Provides a function for generating a random sample from a specified probability distribution. 

[logger.py](./logger.py): Implements a `Logger` class for basic logging functionalities. The class can log messages, warnings, and errors to a specified file.

[test_result_logger.py](./test_result_logger.py): A specialized logger for logging the results of unittest executions. It logs detailed outcomes of tests (to file named 'test_results.log') and simplified pass/fail information (to file named 'test_results_simple.log'), assisting in test result analysis.

[parse_and_repeat.py](./parse_and_repeat.py): Parses strings representing lists, evaluates the elements as stringliterals and repeats based on the specified pattern. 

[search_files_with_name.py](./search_files_with_name.py): Provides a function for searching files with a specific name in a directory and its subdirectories. 

[simple_popup_handler.py](./simple_popup_handler.py): A utility class for displaying various simple pop-up dialogs using Tkinter. It can show informational messages, prompt for user inputs, and facilitate file selections.

[copy_json_exclude_entries.py](./copy_json_exclude_entries.py): Provides a function for copying a JSON file while excluding specified entries.

