import os
import subprocess


def execute_python_script(script_path, env_python_path):
    """
    Executes a Python script located at the specified path using the Python
    interpreter from the virtual environment and captures its output.

    Args:
        - script_path (str): The absolute path to the Python script to
            execute.
        - env_python_path (str): The path to the Python interpreter in the
            virtual environment.

    Returns:
        - str: The output from the script execution or an error message if
            execution fails.
    """
    python_path = os.path.join(env_python_path, "bin", "python")
    try:
        completed_process = subprocess.run(
            [python_path, "-u", script_path],
            capture_output=True,
            text=True,
            check=True,
        )
        return completed_process.stdout
    except subprocess.CalledProcessError as e:
        return (
            f"Error running script: {e}\nOutput: {e.output}\nError Output: {e.stderr}"
        )


if __name__ == "__main__":
    script_path = "/path/to/python/script.py"
    env_python_path = "path/to/python/interpreter"
    script_output = execute_python_script(script_path, env_python_path)
    with open("output.txt", "w") as output_file:
        output_file.write(script_output)
