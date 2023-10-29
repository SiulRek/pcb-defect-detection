import pandas as pd
import os
import xml.etree.ElementTree as ET 

ROOT_DIRECTORY = os.path.curdir

PATH_ANNOTATIONS = os.path.join(ROOT_DIRECTORY, "data\pcb_defects_kaggle\Annotations") # Path to annotations.
PATH_IMAGE = os.path.join(ROOT_DIRECTORY, "data\pcb_defects_kaggle\images") # Path to .jpg images.

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
    for path, subdirs, files in os.walk(path_an):
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
                            dataset['category_codes'] += [1]
                        elif name == 'mouse_bite':
                            dataset['category_codes'] += [2]
                        elif name == 'open_circuit':
                            dataset['category_codes'] += [3]
                        elif name == 'short':
                            dataset['category_codes'] += [4]
                        elif name == 'spur':
                            dataset['category_codes'] += [5]
                        elif name == 'spurious_copper':
                            dataset['category_codes'] += [6]
                        
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

    for key in dataset.keys():
        print(key, len(dataset[key]))

    dataframe = pd.DataFrame(dataset)

    if create_annotation_summary:
        file_path = os.path.join(path_an, 'annotation_summary.csv')
        dataframe.to_csv(file_path, sep=';')


    return dataframe


if __name__ == '__main__':
    get_dataframe()