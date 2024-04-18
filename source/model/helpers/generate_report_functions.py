
import os
import pickle


def get_experiments_data(project_directory, pickle_filename):
    """ 
    Get the data from all experiments in the project directory.

    Args:
        project_directory (str): The root directory of the project.
        pickle_filename (str): The name of the pickle file containing the experiment results.
    
    Returns:
        list: A list of tuples, where each tuple contains the relative path of the experiment 
        directory and the data loaded from the pickle file.
    """
    if not os.path.exists(project_directory):
        raise FileNotFoundError(f"Directory {project_directory} not found.")
    
    experiments_data = []
    for root, _, files in os.walk(project_directory):
        if pickle_filename in files:
            pickle_path = os.path.join(root, pickle_filename)
            with open(pickle_path, 'rb') as file:
                data = pickle.load(file)
            relative_path = os.path.relpath(root, project_directory)
            experiments_data.append((relative_path, data))

    if not experiments_data:
        raise FileNotFoundError(f"No experiments found in {project_directory}.")
    return experiments_data


def sort_experiments_data(experiments_data, sort_criteria):
    """ 
    Sort the experiments data based on the sort criteria.

    Args:
        experiments_data (list): A list of tuples where each tuple contains the relative path of the
            experiment directory and the data loaded from the pickle file.
        sort_criteria (tuple): Specifies the parameter to sort result by. 
            The first element is the name of the metrics and the second element is the name of the 
            statistics.
    
    Returns:
        list: A sorted list of tuples where each tuple contains the relative path of the experiment 
            directory and the data loaded from the pickle file.
    """
    metric, statistic = sort_criteria
    try:
        sorted_data = sorted(experiments_data, key=lambda x: x[1][statistic][metric], reverse=True)
        return sorted_data
    except KeyError as e:
        for path, data in experiments_data:
            if statistic not in data:
                msg = f"Statistic {statistic} not found in experiment {path}."
            elif metric not in data[statistic]:
                msg = f"Metric {metric} not found in experiment {path}."
        raise ValueError(f"{msg}") from e

# def generate_report_elements(project_directory, experiments_data, figure_names, sort_criteria):
#     """
#     Generate report elements as an ordered list of tuples.

#     Args:
#         project_directory (str): The root directory of the project.
#         experiments_data (list): A list of tuples containing the relative path of the 
#             experiment directory and the data loaded from the pickle file.
#         figure_names (list): The names of the figures to be included in the report for each experiment.
#         sort_criteria (tuple): The sorting criteria for the experiments data.
    
#     Returns:
#         list: An ordered list of tuples, where the first element is the component type 
#             ('title', 'table', 'figure', 'text') and the second is the content for that component.
#     """
#     document_components = []

#     # Title of the report
#     document_components.append(('title', '# Experiment Results Report\n'))

#     # Summary table setup
#     headers = ['Statistic'] + [exp_data[0] for exp_data in experiments_data]
#     table_lines = [headers] + [['---'] * len(headers)]

#     # Data for summary table
#     metrics = list(experiments_data[0][1][sort_criteria[1]].keys())
#     for metric in metrics:
#         row = [f'{metric} ({sort_criteria[1]})'] + [f'{exp_data[1][sort_criteria[1]][metric]:.2f}' for exp_data in experiments_data]
#         table_lines.append(row)

#     document_components.append(('table', table_lines))

#     # Individual experiment details
#     for exp_data in experiments_data:
#         document_components.append(('header', f'## Experiment: {exp_data[0]}\n'))

#         # Figures
#         for figure in figure_names:
#             figure_path = os.path.join(project_directory, exp_data[0], figure)
#             if os.path.exists(figure_path):
#                 document_components.append(('figure', f'![{figure}]({figure_path})\n'))
#             else:
#                 document_components.append(('text', f'Figure {figure} not available.\n'))

#         # Detailed statistics table
#         headers = ['Statistic'] + list(exp_data[1]['mean'].keys())
#         table_lines = [headers] + [['---'] * len(headers)]
#         for statistic in exp_data[1].keys():
#             row = [statistic] + [f'{metric:.2f}' for metric in exp_data[1][statistic].values()]
#             table_lines.append(row)
#         document_components.append(('table', table_lines))

#     return document_components




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
    