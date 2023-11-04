import python_code.load_raw_data.kaggle_dataset as kaggle_dataset
import python_code.load_raw_data.deep_pcb_tangali5201 as deep_pcb_dataset

kaggle_dataset.save_tf_record()
deep_pcb_dataset.save_tf_records()

print('Saving of Tensorflow Records was successfull.')