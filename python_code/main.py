from python_code.image_preprocessing.image_preprocessor import ImagePreprocessor
from python_code.model.pcb_defect_detector import PCBDefectDetector

#IMPORTANT ANNOTATIONS: The current functionality serves just as a template and as documentation of the current software plan.


if __name__ == '__main__':

    preprocessing_config = {
        'resize': {'width': 2000, 'height': 2000},
        'enhance_contrast': {'factor': 1.5},
        'reduce_noise': {'method': 'gaussian'}
    }

    train_data, test_data = (None, None)

    preprocessor = ImagePreprocessor(preprocessing_config)
    detector = PCBDefectDetector(preprocessor)
    detector.build_model()
    history = detector.train(train_data)
    detector.predict(test_data)
