import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox


class SimplePopupHandler:
    """
    A handler class for displaying various simple pop-up dialogs using Tkinter.

    This class provides easy-to-use methods for interacting with the user via
    simple dialog boxes. It can display informational messages, ask the user to
    input strings or numbers, present yes/no questions, and facilitate file
    selection.
    """

    def __init__(self):
        """
        Initialize the SimplePopupHandler object.

        This constructor currently does not take any parameters, as all methods
        are designed to be called independently and do not rely on instance
        attributes.
        """
        pass

    def display_popup_message(self, message):
        """
        Display a popup window with the given message.

        Parameters:
            - message (str): The message to be displayed in the popup
                window.
        """
        try:
            messagebox.showinfo("Message", message)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def get_string(self, dialog="Enter a string"):
        """
        Prompt the user to enter a string and display it in a popup window.

        Parameters:
            - dialog (str): The dialog message to display to the user.
                Default is 'Enter a string'.
        """
        try:
            input_str = simpledialog.askstring("Input", dialog)

            if input_str is None:
                self.display_popup_message("Operation canceled by user")

        except Exception as e:
            self.display_popup_message(f"An error occurred: {str(e)}")

    def get_numbers_list_from_input(
        self, dialog="Enter numbers separated by commas:", success_flag=True
    ):
        """
        Prompt the user to enter a string of numbers separated by commas,
        convert it to a list of numbers, and optionally display a 'Success'
        message in a popup window. Handle empty or invalid input.

        Parameters:
            - dialog (str): The dialog message to display to the user.
            - Default is "Enter numbers separated by commas:".
            - success_flag (bool): Whether to display a success message.
                Default is True.

        Returns:
            - numbers_list (list): List of numbers entered by the user.
        """

        try:
            input_str = simpledialog.askstring("Input", dialog)

            if input_str is not None:
                input_list = input_str.split(",")
                numbers_list = []

                for item in input_list:
                    try:
                        number = float(item.strip())
                        numbers_list.append(number)
                    except ValueError:
                        self.display_popup_message(
                            f"Invalid input: '{item}' is not a valid number."
                        )
                        numbers_list = []  # Clear the list in case of an error
                        break

                if not numbers_list:
                    self.display_popup_message(
                        "Empty or invalid input. Please enter valid numbers."
                    )
                else:
                    if len(numbers_list) == 1:
                        # If there's only one number, convert it to a list with a single element
                        numbers_list = [numbers_list[0]]
                    if success_flag:
                        self.display_popup_message(
                            "Input successfully converted to a list."
                        )
            else:
                self.display_popup_message("Operation canceled by user")

        except Exception as e:
            self.display_popup_message(f"An error occurred: {str(e)}")

        return numbers_list

    def ask_yes_no_question(self, question):
        """
        Display a popup window with a Yes/No question and return True if Yes is
        clicked, False if No is clicked.

        Parameters:
            - question (str): The question to be displayed in the popup
                window.

        Returns:
            - bool: True if Yes is clicked, False if No is clicked.
        """
        try:
            response = tk.messagebox.askquestion("Question", question)

            if response == "yes":
                return True
            else:
                return False
        except Exception as e:
            self.display_popup_message(f"An error occurred: {str(e)}")

    def browse_file(self):
        """
        Open a file dialog to let the user browse and select a file, and return
        the selected file path.

        Returns:
            - file_path (str): The path of the selected file.
        """
        try:
            file_path = filedialog.askopenfilename()
            if file_path:
                return file_path
            else:
                self.display_popup_message("No file selected.")
        except Exception as e:
            self.display_popup_message(f"An error occurred: {str(e)}")


# Example of using the class:
if __name__ == "__main__":
    popup_handler = SimplePopupHandler()
    answer = popup_handler.ask_yes_no_question("Is the Python Experience ENDLESS?")
    if answer == True:
        popup_handler.display_popup_message("Yes it is!")
    elif answer == False:
        popup_handler.display_popup_message("No, it is not!")
    else:
        popup_handler.display_popup_message("Oops, something unexpected happened.")
