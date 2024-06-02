""" This module automates the creation of TensorFlow records for datasets available
in the pcb-defect-detection repository. """

import os

import source.load_raw_data.deep_pcb_tangali5201 as deep_pcb_dataset
import source.load_raw_data.kaggle_dataset as kaggle_dataset
from source.utils import SimplePopupHandler
from source.utils import search_files_with_name

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


if __name__ == "__main__":
    popup_handler = SimplePopupHandler()
    try:
        kaggle_dataset.save_tf_record()
        deep_pcb_dataset.save_tf_records()
    except Exception as exc:
        popup_handler.display_popup_message(f"An error occurred: {str(exc)}")
        raise exc

    tf_records_paths = []
    tf_records_paths.append(
        os.path.join(ROOT_DIR, r"data\tensorflow_records\pcb_defects_kaggle.tfrecord")
    )
    tf_records_paths.append(
        os.path.join(ROOT_DIR, r"data\tensorflow_records\deep_pcb_train.tfrecord")
    )
    tf_records_paths.append(
        os.path.join(ROOT_DIR, r"data\tensorflow_records\deep_pcb_test.tfrecord")
    )

    successfully_created = all([os.path.exists(path) for path in tf_records_paths])

    if successfully_created:
        message = "Tensorflow Records have been successfully created!\n"
        test_runner_abs_paths = search_files_with_name(ROOT_DIR, "all_test_runner.py")
        test_runner_paths = [
            "\n - " + absolute_path[len(ROOT_DIR) :]
            for absolute_path in test_runner_abs_paths
        ]
        paths_message = "\nPlease execute the following file to verify the setup:"
        paths_message += "".join(test_runner_paths)
        final_message = f"{message}{paths_message}"
        print(final_message)
        SimplePopupHandler().display_popup_message(final_message)
    else:
        SimplePopupHandler().display_popup_message(
            "Something went wrong while creating Tensorflow Records."
        )
