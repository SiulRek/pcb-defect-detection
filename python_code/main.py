from python_code.image_preprocessing.image_preprocessor import ImagePreprocessor
from python_code.model.pcb_defect_detector import PCBDefectDetector

#IMPORTANT ANNOTATIONS: The current functionality serves just as a template and as documentation of the current software plan.  Simplified Training procedure is shown here.


if __name__ == '__main__':
    pipeline = {
        # Add preprocessing steps here.
    }
    
    model = {
        # Add model layers here.
    }

    train_data, test_data = (None, None)

    preprocessor = ImagePreprocessor()
    preprocessor.set_pipe(pipeline)
    processed_train_data = preprocessor.process(train_data)
    processed_test_data = preprocessor.process(test_data)
    detector = PCBDefectDetector()
    detector.build_model(model)
    history = detector.train(processed_train_data)
    detector.predict(processed_test_data)
