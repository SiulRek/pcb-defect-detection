import pandas as pd
import os
import xml.etree.ElementTree as ET 

from source.load_raw_data.get_tf_dataset import get_tf_dataset_from_df
from source.load_raw_data.dataset_serialization import load_tfrecord_from_file, save_tfrecord_to_file
from source.load_raw_data.category_codes import Category

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')

PATH_ANNOTATIONS = os.path.join(ROOT_DIR, r"data\pcb_defects_kaggle\Annotations") # Path to annotations.
PATH_IMAGE = os.path.join(ROOT_DIR, r"data\pcb_defects_kaggle\images") # Path to .jpg images.
PATH_PCB_USED = os.path.join(ROOT_DIR, r"data\pcb_defects_kaggle\PCB_USED") # Path to .jpg PCB used.
RECORD_FILE = os.path.join(ROOT_DIR, r"data\tensorflow_records\pcb_defects_kaggle.tfrecord")


def get_dataframe(path_an=PATH_ANNOTATIONS, path_im=PATH_IMAGE, create_annotation_summary=True):
    """
    Load and process Kaggle PCB defect dataset annotations and images to create a Pandas DataFrame(https://www.kaggle.com/datasets/akhatova/pcb-defects/data).

    Parameters:
    - path_an (str): Path to the directory containing XML annotations.
    - path_im (str): Path to the directory containing images.
    - create_annotation_summary (bool): If True, create an annotation summary CSV file.

    Returns:
    - pandas.DataFrame: A DataFrame containing processed data.

    This function reads XML annotations for PCB defects, extracts information and stores
    the information in a DataFrame.

    The resulting DataFrame contains the following columns:
    - 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'file', 'defect_x_center', 'defect_y_center',
      'defect_width', 'defect_height', 'image_width', 'image_height', 'width_ratio', 'height_ratio',
      'path', 'category_codes'
      
    Implementation Reference: https://www.kaggle.com/code/leedh000/cpb-defect-detecting-with-yolov7.
    """
    
    dataset = {
                "class":[],
                "xmin":[],
                "ymin":[],   
                "xmax":[],
                "ymax":[],
                "file":[],
                "defect_x_center":[],         # Relative to image_width.
                "defect_y_center":[],       # Relative to image_heigth.
                "defect_width":[],
                "defect_height":[],
                "image_width":[],
                "image_height":[],
                "width_ratio":[],           # Defect width/Image width.
                "height_ratio":[],           # Defect heigth/Image heigth.
                "path":[],
                "category_codes":[]
            }

    all_files = []          # Stores the path to all Files
    for path, _, files in os.walk(path_an):
        #print([path, subdirs, files])
        for name in files:
            if '.csv' not in name:
                all_files.append(os.path.join(path, name))
            
    for anno in all_files:

        tree = ET.parse(anno)
               
        cnt = 0
        for elem in tree.iter():
            if 'size' in elem.tag:
                for attr in list(elem):
                    if 'width' in attr.tag: 
                        image_width = int(round(float(attr.text)))
                    if 'height' in attr.tag:
                        image_height = int(round(float(attr.text)))

            if 'object' in elem.tag:
                cnt += 1

                for attr in list(elem):

                    if 'name' in attr.tag:
                        name = attr.text                 
                        dataset['class']+=[name]
                        
                        if name == 'missing_hole':
                            dataset['category_codes'] += [Category.MISSING_HOLE.value]
                        elif name == 'mouse_bite':
                            dataset['category_codes'] += [Category.MOUSE_BITE.value]
                        elif name == 'open_circuit':
                            dataset['category_codes'] += [Category.OPEN_CIRCUIT.value]
                        elif name == 'short':
                            dataset['category_codes'] += [Category.SHORT.value]
                        elif name == 'spur':
                            dataset['category_codes'] += [Category.SPUR.value]
                        elif name == 'spurious_copper':
                            dataset['category_codes'] += [Category.SPURIOUS_COPPER.value]
                        
                        dataset['image_width']+=[image_width]
                        dataset['image_height']+=[image_height] 
                        dataset['file']+=[anno.split('/')[-1][0:-4]] 
                                
                    if 'bndbox' in attr.tag:
                        xmin = -1       # To avoid assigment of values from the previous object. 
                        ymin = -1
                        xmax = -1
                        ymax = -1
                        for dim in list(attr):
        
                            if 'xmin' in dim.tag:
                                xmin = int(round(float(dim.text)))
                                dataset['xmin']+=[xmin]
                            if 'ymin' in dim.tag:
                                ymin = int(round(float(dim.text)))
                                dataset['ymin']+=[ymin]                                
                            if 'xmax' in dim.tag:
                                xmax = int(round(float(dim.text)))
                                dataset['xmax']+=[xmax]                                
                            if 'ymax' in dim.tag:
                                ymax = int(round(float(dim.text)))
                                dataset['ymax']+=[ymax]   
                        
                        width = xmax - xmin
                        height = ymax - ymin
                        dataset['defect_width'] += [width]
                        dataset['defect_height'] += [height]
                        dataset['defect_x_center'] += [(((xmin + width) + xmin) / 2) / image_width]

                        if((((xmin + width) + xmin) / 2) / image_width >= 1):
                            print(xmin, ' + ', width, ' + ', xmin, ' / ', 2, ' / ', image_width)

                        dataset['defect_y_center'] += [(((ymin + height) + ymin) / 2) / image_height]
                        dataset['width_ratio'] += [width / image_width]
                        dataset['height_ratio'] += [height / image_height]                             

        for i in range(cnt):
            tmp_path = tree.find('path').text
            dataset['path']+= [path_im + tmp_path[32:]]     # Adapt path to pcb-defect-detection Workspace     

    dataframe = pd.DataFrame(dataset)

    if create_annotation_summary:
        file_path = os.path.join(path_an, 'annotation_summary.csv')
        dataframe.to_csv(file_path, sep=';')
        
    return dataframe


def get_tf_dataset(path_an=PATH_ANNOTATIONS, path_im=PATH_IMAGE, create_annotation_summary=False, random_seed=75, sample_num=-1):
    """
    Generates a TensorFlow Dataset from the Kaggle PCB defects dataset.
    
    Parameters:
    - path_an (str): Path to the directory containing XML annotations.
    - path_im (str): Path to the directory containing images.
    - create_annotation_summary (bool): If True, create an annotation summary CSV file.
    - random_seed (int, optional): The random seed for shuffling the dataset. Defaults to 34.
    - sample_num (int, optional): Numbers of samples to take from the dataframe. Defaults to -1 -> All Samples are taken.
    
    Returns:
    - tf.data.Dataset: A TensorFlow Dataset containing tuples of (image, category_code),
        where 'image' is the decoded image file and 'category_code' is an integer label.
    """
    
    df = get_dataframe(path_an, path_im, create_annotation_summary)

    return get_tf_dataset_from_df(df, random_seed=random_seed, sample_num=sample_num)


def get_tf_dataset_with_category_zero(image_path=PATH_PCB_USED, random_seed=75):
    """
    Generates a TensorFlow Dataset from the Kaggle PCB defects dataset with category zero.

    Parameters:
    - image_path (str): Path to the directory containing images.
    - random_seed (int, optional): The random seed for shuffling the dataset. Defaults to 34.
    """
    all_files = []       
    for image_name in os.listdir(image_path):
        if '.JPG' in image_name:
            all_files.append(os.path.join(image_path, image_name))
    categories = [0] * len(all_files)
    df_zero_category = pd.DataFrame({'path': all_files, 'category_codes': categories})
    return get_tf_dataset_from_df(df_zero_category, random_seed=random_seed)


def get_tf_datasets_for_each_category(path_an=PATH_ANNOTATIONS, path_im=PATH_IMAGE, create_annotation_summary=False, random_seed=75):
    """
    Generates TensorFlow Datasets for each category of the Kaggle PCB defects dataset.

    Parameters:
    - path_an (str): Path to the directory containing XML annotations.
    - path_im (str): Path to the directory containing images.
    - create_annotation_summary (bool): If True, create an annotation summary CSV file.
    - random_seed (int, optional): The random seed for shuffling the dataset. Defaults to 34.

    Returns:
    - dict: A dictionary containing TensorFlow Datasets for each category.
    """
    
    df = get_dataframe(path_an, path_im, create_annotation_summary)
    datasets = {}
    for category in Category:
        category_df = df[df['category_codes'] == category.value]
        if category_df.empty:
            continue
        datasets[category.name] = get_tf_dataset_from_df(category_df, random_seed=random_seed)
    return datasets

def save_tf_record():
    """  Save the Kaggle PCB defects dataset as a TFRecord file. 

    This function generates a TensorFlow Dataset from the Kaggle PCB defects dataset and saves it to a TFRecord file for efficient future access. It includes image data and corresponding category codes.

    Note:
        The TFRecord is saved to the path specified in 'RECORD_FILE'.
        An annotation summary CSV file is also generated in the process.
    """
    save_tfrecord_to_file(get_tf_dataset(create_annotation_summary=True), RECORD_FILE)


def load_tf_record():
    """  Load the TensorFlow dataset from a TFRecord file.

    This function specifically loads the dataset from a TFRecord file that corresponds to the Kaggle PCB defects dataset. It ensures that the dataset is parsed and optimized for use, containing images along with their corresponding targets.

    Returns:
    - tf.data.Dataset: A TensorFlow Dataset containing tuples of (image, category_code),
        where 'image' is the decoded image file and 'category_code' is an integer label.
    """
    return load_tfrecord_from_file(RECORD_FILE)




if __name__ == '__main__':
    # save_tf_record()
    get_tf_datasets_for_each_category()