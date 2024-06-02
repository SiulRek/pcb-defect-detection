import random

import numpy as np

import cv2
from source.preprocessing.helpers.for_steps.step_base import StepBase


class DilateErodeSequencer(StepBase):
    """
    A preprocessing step that applies a sequence of dilation and erosion
    operations to an image. This class can automatically generate an operation
    sequence based on the provided iterations and erosion probability.
    """

    arguments_datatype = {
        "kernel_size": int,
        "sequence": str,
        "iterations": int,
        "erosion_probability": float,
    }
    name = "Dilate Erode Sequencer"

    def __init__(
        self, kernel_size=3, sequence="de", iterations=-1, erosion_probability=0.5
    ):
        """
        Initializes the DilateErodeSequencer object. If iterations are positive
        and the erosion probability is within a valid range (0 to 1), it
        automatically creates an operation sequence combining dilation and
        erosion.

        Args:
            - kernel_size (int): Size of the kernel for dilation/erosion.
            - sequence (str): The sequence of operations ('d' for dilation,
                'e' for erosion).
            - iterations (int): Number of times the sequence is repeated.
            - erosion_probability (float): Probability of choosing erosion
                in the random sequence generation.
        """
        if not 0 <= erosion_probability <= 1:
            msg = "Erosion probability must be between 0 and 1."
            raise ValueError(msg)

        sequence = self.generate_sequence(sequence, iterations, erosion_probability)

        parameters = {
            "kernel_size": kernel_size,
            "sequence": sequence,
            "iterations": iterations,
            "erosion_probability": erosion_probability,
        }

        super().__init__(parameters)

    def generate_sequence(self, sequence, iterations, erosion_probability):
        """
        Generates a sequence of operations based on the specified probability
        and iterations.

        Args:
            - sequence (str): Initial sequence of operations.
            - iterations (int): Number of iterations to extend the sequence.
            - erosion_probability (float): Probability of choosing erosion.

        Returns:
            - str: The generated sequence of operations.
        """
        if iterations > 1:
            operations = [
                self._choose_operation(erosion_probability) for _ in range(iterations)
            ]
            random_sequence = "".join(operations)
            return random_sequence
        return sequence

    def _choose_operation(self, erosion_probability):
        """
        Randomly chooses between dilation ('d') and erosion ('e') based on the
        specified probability.

        Args:
            - erosion_probability (float): Probability of choosing erosion.

        Returns:
            - str: 'd' for dilation or 'e' for erosion.
        """
        return "e" if random.random() < erosion_probability else "d"

    @StepBase._nparray_pyfunc_wrapper
    def process_step(self, image_nparray):
        """
        Applies the preprocessing steps to the given image.

        Args:
            - image_nparray (np.array): The image to be processed.

        Returns:
            - np.array: The processed image.
        """
        kernel = np.ones(
            (self.parameters["kernel_size"], self.parameters["kernel_size"]), np.uint8
        )
        processed_image = image_nparray

        for operation in self.parameters["sequence"]:
            if operation == "d":
                processed_image = cv2.dilate(
                    processed_image, kernel, iterations=self.parameters["iterations"]
                )
            elif operation == "e":
                processed_image = cv2.erode(
                    processed_image, kernel, iterations=self.parameters["iterations"]
                )
            else:
                msg = "Invalid operation in sequence."
                msg += "Only 'd' (dilation) and 'e' (erosion) are allowed."
                raise ValueError(msg)

        return processed_image


if __name__ == "__main__":
    step = DilateErodeSequencer()
    print(step.get_step_json_representation())
