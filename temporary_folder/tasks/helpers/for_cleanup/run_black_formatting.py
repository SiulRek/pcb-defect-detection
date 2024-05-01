import subprocess

def format_with_black(script_path, env_python_path):
    """
    Formats a Python script located at the specified path using black, run from the Python interpreter in the virtual environment.
    
    Args:
        script_path (str): The absolute path to the Python script to be formatted.
        env_python_path (str): The path to the Python interpreter in the virtual environment.

    Returns:
        str: The result of the formatting operation or an error message if the operation fails.
    """
    try:
        completed_process = subprocess.run(
            [env_python_path, "-m", "black", script_path],
            capture_output=True,
            text=True,
            check=True
        )
        return completed_process.stdout if completed_process.stdout else "No changes made."
    except subprocess.CalledProcessError as e:
        return f"Formatting error: {e}\nOutput: {e.stdout}\nError Output: {e.stderr}"

if __name__ == "__main__":
    script_path = "/path/to/python/script.py"
    env_python_path = "/path/to/venv/bin/python"
    formatting_result = format_with_black(script_path, env_python_path)
    print(formatting_result)