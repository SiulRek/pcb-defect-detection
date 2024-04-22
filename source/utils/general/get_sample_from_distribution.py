import numpy as np


def get_sample_from_distribution(distribution_data):
    """
    Generates a single random sample from a specified probability distribution.

    This function dynamically selects a distribution function based on the input 
    dictionary 'distribution_data', which should contain the key 'distribution' 
    specifying the type of distribution, and additional parameters required by 
    the selected distribution.

    Parameters:
    distribution_data (dict): A dictionary containing the distribution type 
                              and its corresponding parameters. 

    Returns:
    float: A single sample from the specified distribution.
    """

    distribution_map = {
    'gaussian': np.random.normal, # mean ('loc'), standard deviation ('scale')
    'uniform': np.random.uniform,  # lower boundary ('low'), upper boundary ('high')
    'exponential': np.random.exponential, # scale ('scale' or '1/lambda')
    'poisson': np.random.poisson,  # rate ('lam' or expected number of occurrences)
    'binomial': np.random.binomial,  # number of trials ('n'), probability of success ('p')
    'gamma': np.random.gamma,  # shape parameter ('shape'), scale parameter ('scale')
    'beta': np.random.beta,  # alpha parameter ('a'), beta parameter ('b')
    'lognormal': np.random.lognormal,  # mean ('mean'), standard deviation ('sigma') of the underlying normal distribution
    'laplace': np.random.laplace # location parameter ('loc'), scale parameter ('scale')
    }

    if 'distribution' not in distribution_data.keys():
        raise KeyError("No key named 'distribution' in 'distribution_data'.")
    
    dist_function = distribution_map.get(distribution_data['distribution'].lower())
    if not dist_function:
        raise ValueError(f"Distribution {distribution_data['distribution']} is not supported.")

    args = {k: v for k, v in distribution_data.items() if k != 'distribution'}
    args['size'] = 1

    try:
        sample = float(dist_function(**args)[0])
    except TypeError as e:
        error_message = "Impossible to call the specified distribution function with contents of 'distribution_data' dict."
        raise ValueError(error_message) from e
     
    return sample
