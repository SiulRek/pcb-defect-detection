import os
import subprocess


def format_with_black(script_path, environment_path):
    """
    Formats a Python script using the Black code formatter.

    Args:
        - script_path (str): The path to the Python script to format.
        - environment_path (str): The path to the virtual environment
            containing the Black package.
    
    Returns:
        - str: The formatted code or an error message if formatting fails.
    """
    python_executable = os.path.join(
        environment_path, "bin" if os.name == "posix" else "Scripts", "python"
    )
    black_script = os.path.join(
        environment_path, "bin" if os.name == "posix" else "Scripts", "black"
    )

    black_command = [black_script, script_path]
    try:
        result = subprocess.run(
            [python_executable] + black_command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Formatting error: {e}\nOutput: {e.stdout}\nError Output: {e.stderr}"


if __name__ == "__main__":
    script_path = "/path/to/python/script.py"
    env_python_path = "/path/to/venv/bin/python"
    formatted_code = format_with_black(script_path, env_python_path)
