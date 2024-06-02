import os
import pickle

from source.model.definitions.general import REPORT_ELEMENT as ELEMENT


def get_experiments_data(project_directory, pickle_filename):
    """
    Get the data from all experiments in the project directory.

    Args:
        - project_directory (str): The root directory of the project.
        - pickle_filename (str): The name of the pickle file containing the
            experiment results.

    Returns:
        - list: A list of tuples, where each tuple contains the relative
            path of the experiment directory and the data loaded from the pickle
            file.
    """
    if not os.path.exists(project_directory):
        msg = f"Directory {project_directory} not found."
        raise FileNotFoundError(msg)

    experiments_data = []
    for root, _, files in os.walk(project_directory):
        if pickle_filename in files:
            pickle_path = os.path.join(root, pickle_filename)
            with open(pickle_path, "rb") as file:
                data = pickle.load(file)
            relative_path = os.path.relpath(root, project_directory)
            experiments_data.append((relative_path, data))

    if not experiments_data:
        msg = f"No experiments found in {project_directory}."
        raise FileNotFoundError(msg)
    return experiments_data


def sort_experiments_data(experiments_data, sort_criteria):
    """
    Sort the experiments data based on the sort criteria.

    Args:
        - experiments_data (list): A list of tuples where each tuple
            contains the relative path of the experiment directory and the data
            loaded from the pickle file.
        - sort_criteria (tuple): Specifies the parameter to sort result by.
            The first element is the name of the metrics and the second element
            is the name of the statistics.

    Returns:
        - list: A sorted list of tuples where each tuple contains the
            relative path of the experiment directory and the data loaded from
            the pickle file.
    """
    metric, statistic = sort_criteria
    try:
        sorted_data = sorted(
            experiments_data, key=lambda x: x[1][statistic][metric], reverse=True
        )
        return sorted_data
    except KeyError as e:
        for path, data in experiments_data:
            if statistic not in data:
                msg = f"Statistic {statistic} not found in experiment {path}."
            elif metric not in data[statistic]:
                msg = f"Metric {metric} not found in experiment {path}."
        msg = f"{msg}"
        raise ValueError(msg)


def make_experiment_name(relative_path):
    names = relative_path.split(os.sep)[::-1]
    title = "_".join(names[:])
    return title


def get_figure_path(project_directory, relative_path, figure_name):
    dir = os.path.join(project_directory, relative_path)
    for file in os.listdir(dir):
        if file.startswith(figure_name):
            return os.path.join(dir, file)
    return None


def generate_report_elements(
    project_directory, experiments_data, header, figure_names, summary_metric
):
    """
    Generate report elements as an ordered list of tuples.

    Args:
        - project_directory (str): The root directory of the project.
        - experiments_data (list): A list of tuples containing the relative
            path of the experiment directory and the data loaded from the pickle
            file.
        - figure_names (list): The names of the figures to be included in
            the report for each experiment.
        - no_name (str): The metric to be used as the primary in summary
            table.

    Returns:
        - list: An ordered list of tuples, where the first element is the
            component type ('title', 'table', 'figure', 'text') and the second
            is the content for that component.
    """
    document_components = []
    document_components.append((ELEMENT.HEADER, header))
    document_components.append((ELEMENT.TITLE, "Result Summary"))
    summary_table = {}
    for path, data in experiments_data:
        experiment_name = make_experiment_name(path)
        summary_table[experiment_name] = {}
        for statistic in data:
            summary_table[experiment_name][statistic] = data[statistic][summary_metric]
    document_components.append((ELEMENT.TABLE, summary_table))
    document_components.append((ELEMENT.TITLE, "Experiment Results"))

    for experiment_data in experiments_data:
        experiment_name = make_experiment_name(experiment_data[0])
        document_components.append((ELEMENT.SUBTITLE, experiment_name))
        for figure in figure_names:
            figure_path = get_figure_path(project_directory, experiment_data[0], figure)
            if figure_path:
                document_components.append((ELEMENT.FIGURE, figure_path))
            else:
                red_text = f"Figure {figure} not available in {experiment_name}."
                document_components.append((ELEMENT.RED_TEXT, red_text))
        document_components.append((ELEMENT.TABLE, experiment_data[1]))
        link = ("link to experiment", experiment_data[0])
        document_components.append((ELEMENT.LINK, link))
    return document_components


# def generate_report(project_directory, pickle_filename, sort_criteria, figure_names, report_filename):
#     """
#     Generate a report in the form of a .md file.

#     Args:
#         project_directory (str): The root directory of the project.
#         pickle_filename (str): The name of the pickle file containing the experiment results.
#         sort_criteria (tuple): Specifies the parameter to sort result by.
#             The first element is the name of the metrics and the second element is the name of the
#             statistics.
#         figure_names (list): The names of the figures to be included in the report for each
#             experiment.
#         report_filename (str): The name of the report file.
#     """
#     experiments_data = get_experiments_data(project_directory, pickle_filename)
#     sorted_experiments_data = sort_experiments_data(experiments_data, sort_criteria)
