import tensorflow as tf


class LabelManager:
    """
    Manages different types of label encoding for machine learning models.

    Attributes:
        - label_type (str): The type of labels managed, which determines the
            encoding method used.
        - num_classes (int, optional): The number of classes used for
            categorical label encoding.

    Methods:
        - encode_label: Depending on the `label_type`, it delegates to the
            corresponding method to encode labels.
    """

    default_label_dtype = {
        "binary": tf.float32,
        "category_codes": tf.float32,
        "sparse_category_codes": tf.float32,
        "object_detection": tf.float32,
    }

    def __init__(self, label_type, num_classes=None, dtype=None):
        """
        Initializes the LabelManager with a specific label encoding type and,
        optionally, the number of classes.

        Args:
            - label_type (str): The type of label encoding to manage.
                Supported types are 'binary', 'category_codes',
                'sparse_category_codes', and 'object_detection'.
            - num_classes (int, optional): The total number of classes.
                Required if label_type is 'category_codes'.
            - dtype (tf.DType, optional): The data type of the label.
        """
        self.label_type = label_type
        self.num_classes = num_classes
        self._encode_label_func = self._get_label_encoder(label_type)
        self.label_dtype = dtype or self.default_label_dtype[label_type]

    def _get_label_encoder(self, label_type):
        """
        Returns the label encoder method based on the label type.

        Args:
            - label_type (str): The type of label encoding to manage.
                Supported types are 'binary', 'category_codes',
                'sparse_category_codes', and 'object_detection'.

        Returns:
            - function: The label encoder method based on the label type.
        """
        if label_type == "binary":
            return self.encode_binary_label
        if label_type == "category_codes":
            return self.encode_categorical_label
        if label_type == "sparse_category_codes":
            return self.encode_sparse_categorical_label
        if label_type == "object_detection":
            return self.encode_object_detection_label
        msg = f"The label type '{label_type}' is not supported."
        raise ValueError(msg)

    def encode_label(self, label):
        """
        Encodes a label based on the label type specified during initialization.

        Args:
            - label (int): The label to encode.

        Returns:
            - tf.Tensor: A TensorFlow constant of the encoded label.
        """
        return self._encode_label_func(label)

    def encode_binary_label(self, label):
        """
        Encodes a binary label into a format suitable for binary classification.

        Args:
            - label (int): The label to encode.

        Returns:
            - tf.Tensor: A TensorFlow constant of the label in binary
                format.
        """
        try:
            label = int(label)
            if label not in [0, 1]:
                msg = "The label is invalid for binary classification."
                raise ValueError(msg)
            label = tf.constant(label, dtype=self.label_dtype)
            return label
        except ValueError as e:
            msg = "The label should be convertible to an integer."
            raise ValueError(msg) from e

    def encode_categorical_label(self, label):
        """
        Encodes a categorical label into one-hot encoded format.

        Args:
            - label (int): The label to encode.

        Returns:
            - tf.Tensor: A one-hot encoded TensorFlow constant of the label.
        """
        try:
            label = int(label)
            label = tf.constant(label, dtype=tf.int8)
            label = tf.one_hot(label, self.num_classes)
            label = tf.cast(label, self.label_dtype)
            return label
        except ValueError as e:
            msg = "The label should be convertible to an integer."
            raise ValueError(msg) from e
        except tf.errors.OpError as e:
            msg = "The number of classes is probably invalid."
            raise ValueError(msg) from e

    def encode_sparse_categorical_label(self, label):
        """
        Encodes a categorical label into sparse format suitable for sparse
        categorical crossentropy loss.

        Args:
            - label (int): The label to encode.

        Returns:
            - tf.Tensor: A TensorFlow constant of the label in sparse
                format.
        """
        try:
            label = int(label)
            label = tf.constant(label, dtype=self.label_dtype)
            return label
        except ValueError as e:
            msg = "The label should be convertible to an integer."
            raise ValueError(msg) from e

    def encode_object_detection_label(self, _):
        """
        Stub method for future implementation of object detection label
        encoding.

        Args:
            - label (int): The label to encode.

        Raises:
            - NotImplementedError: Indicates that the method is not yet
                implemented.
        """
        msg = "Object Detection Labels are not yet implemented."
        raise NotImplementedError(msg)
