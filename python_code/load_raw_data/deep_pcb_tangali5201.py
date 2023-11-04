import pandas as pd
import os

from python_code.load_raw_data.get_tf_dataset import get_tf_dataset_from_df
from python_code.load_raw_data.dataset_serialization import load_tfrecord_from_file, save_tfrecord_to_file


ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
DATASET_DIR = os.path.join(ROOT_DIR, 'data', 'deep_pcb_tangsali5201')
PATH_TRAIN_RECORD = os.path.join(ROOT_DIR, r"data\tensorflow_records\deep_pcb_train.tfrecord")
PATH_TEST_RECORD = os.path.join(ROOT_DIR, r"data\tensorflow_records\deep_pcb_test.tfrecord")

class LoadDataError(Exception): pass

def get_dataframes(dataset_dir=DATASET_DIR, create_annotation_summary=True):
    """Generates and returns pandas DataFrames for training and testing deepPCB datasets of tangali5201 (https://github.com/tangsanli5201/DeepPCB/tree/master).

    Parameters:
        dataset_dir (str, optional): Directory path containing the dataset files.
                                     Defaults to DATASET_DIR.
        create_annotation_summary (bool, optional): If True, writes a summary CSV file. 
                                                    Defaults to True.

    Returns:
        tuple: A tuple containing the training and testing pandas DataFrames.

    The resulting DataFrame contains the following columns:
    - path', 'path_without', 'group', 'xmin', 'ymin', 'xmax', 'ymax', and 'category_codes'.
    """

    train_dataset = get_dataset(os.path.join(dataset_dir, 'trainval.txt'))
    test_dataset = get_dataset(os.path.join(dataset_dir, 'test.txt'))

    train_df = pd.DataFrame(train_dataset)
    test_df = pd.DataFrame(test_dataset)

    if create_annotation_summary:
        file_path = os.path.join(DATASET_DIR, 'annotation_summary.csv')
        df = pd.concat([train_df, test_df])
        df.to_csv(file_path, sep=';')


    return train_df, test_df

def get_dataset(path_to_paths_file, dataset_dir=DATASET_DIR):
    """  Reads dataset information from a given file path and returns a dictionary.

    Parameters:
        path_to_paths_file (str): The path to the file containing paths to images and annotations.
        dataset_dir (str, optional): The directory containing the dataset. Defaults to DATASET_DIR.

    Returns:
        dict: A dictionary containing dataset fields like 'path', 'path_without', 'group',
              'xmin', 'ymin', 'xmax', 'ymax', and 'category_codes'.
    """

    dataset = {
                "path":[],
                "path_without":[],  # Path to image without defects.
                "group":[],   
                "xmin":[],
                "ymin":[],   
                "xmax":[],
                "ymax":[],
                "category_codes":[]
            }

    lines = []
    try:
        with open(path_to_paths_file, 'r') as file:
            lines.extend(file.read().splitlines())
    except FileNotFoundError as e:
        raise LoadDataError(f'{path_to_paths_file} not found.') from e

    for line in lines:    

        paths = line.split(' ') 
        if len(paths) != 2:
            raise LoadDataError(f'The Format of {path_to_paths_file} is incorrect.')
        
        paths = [os.path.join(dataset_dir, path) for path in paths]

        img_path = paths[0].replace('.jpg', '_test.jpg')
        tmp_path = paths[0].replace('.jpg', '_temp.jpg')
        group = os.path.basename(os.path.dirname(os.path.dirname(paths[0])))

        rows = []
        try:
            with open(paths[1], 'r') as file:
                rows = file.read().splitlines()
        except FileNotFoundError as e:
            raise LoadDataError(f'{paths[1]} not found.') from e
        
        for row in rows:

            parts = row.split(' ')

            if len(parts) != 5:
                raise LoadDataError(f'The format of {paths[1]} is incorrect.')
            
            dataset['path'].append(img_path) 
            dataset['path_without'].append(tmp_path) 
            dataset['group'].append(group) 

            x_min = int(parts[0])
            y_min = int(parts[1])
            x_max = int(parts[2])
            y_max  = int(parts[3])

            if x_max < x_min:
                raise LoadDataError(f"Xmax value ({x_max}) not valid (Xmax < Xmin).")
            if y_max < y_min:
                raise LoadDataError(f"Ymax value ({y_max}) not valid (Ymax < Ymin).")

            dataset['xmin'].append(x_min)  
            dataset['ymin'].append(y_min)  
            dataset['xmax'].append(x_max)  
            dataset['ymax'].append(y_max)  
            
            code = int(parts[4])  
            if code == 1:
                dataset['category_codes'].append(3)  
            elif code == 2:
                dataset['category_codes'].append(4)  
            elif code == 3:
                dataset['category_codes'].append(2)  
            elif code == 4:
                dataset['category_codes'].append(5)  
            elif code == 5:
                dataset['category_codes'].append(6) 
            elif code == 6:
                dataset['category_codes'].append(7) 
            else:
                raise LoadDataError(f'Code {code} is invalid.')
                
    return dataset

def get_tf_datasets(dataset_dir=DATASET_DIR, create_annotation_summary=False, random_seed=89):
    """
    Creates a TensorFlow Dataset from Kaggle Dataset.
    
    Parameters:
    - dataset_dir (str): Path to the dataset directory.
    - create_annotation_summary (bool): If True, create an annotation summary CSV file.
    - random_seed (int, optional): The random seed for shuffling the dataset. Defaults to 34.
    
    Returns:
    - train_tf_dataset, test_df_dataset (tf.data.Dataset, tf.data.Dataset):  A tuple of two TensorFlow Dataset object containing shuffled paths and corresponding targets.
    """
    
    train_df, test_df = get_dataframes(dataset_dir, create_annotation_summary)

    train_tf_dataset = get_tf_dataset_from_df(train_df, random_seed)
    test_tf_dataset = get_tf_dataset_from_df(test_df, random_seed)

    return train_tf_dataset, test_tf_dataset

def save_tf_records():
    """    Generates and saves TensorFlow dataset to the TFRecord file.
    """
    train_tf_dataset, test_tf_dataset = get_tf_datasets()
    save_tfrecord_to_file(train_tf_dataset, PATH_TRAIN_RECORD)
    save_tfrecord_to_file(test_tf_dataset, PATH_TEST_RECORD)

def load_tf_records():
    """    Loads the specific TFRecord file and returns a parsed TensorFlow dataset.

    Returns:
    - tf.data.Dataset: A parsed and optimized TensorFlow dataset containing shuffled paths and corresponding targets.
    """
    return load_tfrecord_from_file(PATH_TRAIN_RECORD, PATH_TEST_RECORD)


if __name__ == '__main__':
    get_tf_datasets()




            



    