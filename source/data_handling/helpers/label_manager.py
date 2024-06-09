import tensorflow as tf


class LabelManager:
    """
    Manages different types of label encoding for machine learning models.

    Attributes:
        - category_names (list): The existing category names for label
            encoding.
        - num_categories (int): The number of categories used for
            categorical label encoding.

    Methods:
        - encode_label: Depending on the `label_type`, it delegates to the
            corresponding method to encode labels.
        - decode_label: Decodes a label from a numeric format to a string
            format.
    """

    default_label_dtype = {
        "binary": tf.float32,
        "category_codes": tf.float32,
        "sparse_category_codes": tf.float32,
        "object_detection": tf.float32,
    }

    def __init__(self, label_type, category_names=None, dtype=None):
        """
        Initializes the LabelManager with a specific label encoding type and,
        optionally, the number of categories.

        Args:
            - label_type (str): The type of label encoding to manage.
                Supported types are 'binary', 'category_codes',
                'sparse_category_codes', and 'object_detection'.
            - category_names (list, optional): The existing category names
                for label encoding.
            - dtype (tf.DType, optional): The data type of the label.
        """
        self._label_type = label_type
        self.num_categories = None
        self.category_names = None
        self._set_category_params(category_names)
        self._encode_label_func = self._get_label_encoder(label_type)
        self._label_dtype = dtype or self.default_label_dtype[label_type]

    @property
    def label_type(self):
        """
        Returns the label type of the manager.

        Returns:
            - str: The label type of the manager.
        """
        return self._label_type

    @property
    def label_dtype(self):
        """
        Returns the data type of the label.

        Returns:
            - tf.DType: The data type of the label.
        """
        return self._label_dtype

    def _set_category_params(self, category_names):
        """
        Sets the number of categories based on the category names provided.

        Args:
            - category_names (list): The list of category names.
        """
        if not category_names and self._label_type in [
            "category_codes",
            "sparse_category_codes",
        ]:
            msg = "The category names are required at least to derive the number of categories."
            raise ValueError(msg)
        elif not category_names and self._label_type == "binary":
            self.num_categories = 2
            self.category_names = ["0", "1"]
        elif category_names:
            self.category_names = category_names
            self.num_categories = len(category_names)

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

    def convert_to_numeric(self, label):
        """
        Converts a label to a numeric format in the case of string labels. If it
        is not a string, it returns the label as is.

        Args:
            - label (str or int): The label to convert.

        Returns:
            - int: The label in numeric format.
        """
        if not isinstance(label, str):
            return label
        try:
            return int(label)
        except ValueError:
            pass
        try:
            return self.category_names.index(label)
        except ValueError as e:
            msg = "The label is not in the list of category names."
            raise ValueError(msg) from e

    def encode_label(self, label):
        """
        Encodes a label based on the label type and category names specified
        during initialization to a tensor format.

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
            label = self.convert_to_numeric(label)
            if label not in [0, 1]:
                msg = "The label is invalid for binary classification."
                raise ValueError(msg)
            label = tf.constant(label, dtype=self._label_dtype)
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
            label = self.convert_to_numeric(label)
            label = tf.constant(label, dtype=tf.int8)
            label = tf.one_hot(label, self.num_categories)
            label = tf.cast(label, self._label_dtype)
            return label
        except ValueError as e:
            msg = "The label should be convertible to an integer."
            raise ValueError(msg) from e
        except tf.errors.OpError as e:
            msg = "The number of categories is probably invalid."
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
            label = self.convert_to_numeric(label)
            label = tf.constant(label, dtype=self._label_dtype)
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

    def decode_label(self, label):
        """
        Decodes a label from a numeric format to a string format based on the
        category names provided during initialization.

        Args:
            - label (int): The label to decode.

        Returns:
            - str: The label in string format.
        """
        if not self.category_names:
            msg = "No category names are provided for conversion."
            raise ValueError(msg)
        if not isinstance(label, int) and not isinstance(label, tf.Tensor):
            msg = "The label is not a numeric label."
            raise ValueError(msg)
        try:
            return self.category_names[label]
        except IndexError as e:
            msg = "The label is not in the list of category names."
            raise ValueError(msg) from e
