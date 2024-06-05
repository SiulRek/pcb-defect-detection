import os

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def extract_test_errors(file_path):
    current_test_title_line = None
    errors = []

    with open(file_path, "r") as file:
        lines = file.readlines()

    for line in lines:
        if "---" in line:
            current_test_title_line = line.strip()

        if "ERROR" in line:
            error_line = line.strip()
            if current_test_title_line:
                errors.append((current_test_title_line, error_line))

    base, ext = os.path.splitext(file_path)
    output_file_name = f"{base}_errors{ext}"
    output_file_path = os.path.join(os.path.dirname(file_path), output_file_name)

    with open(output_file_path, "w") as out_file:
        for title_line, error_line in errors:
            out_file.write(f"{title_line}\n{error_line}\n\n")

    return output_file_path


if __name__ == "__main__":
    log_file = os.path.join(ROOT_DIR, "source/test_results_simple.log")
    errors_file_path = extract_test_errors(log_file)
    print(f"Errors log has been saved to: {errors_file_path}")
