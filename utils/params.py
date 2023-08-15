import itertools
import pandas as pd
import numpy as np


def generate_params(params: dict) -> list:
    """Generate the parameters for the simulation.

    Params:
        params (dict): The dictionary of parameters

    Returns:
        params_list: a list of lists of parameters
    """
    params_list = list(itertools.product(*params.values()))
    params_list = [list(x) for x in params_list]
    return params_list


def test_generate_params(params: dict):
    """Test the generate_params function.

    Params:
        params (dict): The dictionary of parameters
    """
    params_list = generate_params(params)
    print(params_list)


def generate_params_df(params: dict) -> pd.DataFrame:
    """Generate the parameters for the simulation.

    Params:
        params (dict): The dictionary of parameters

    Returns:
        params_df: a dataframe of parameters
    """
    params_list = generate_params(params)
    params_df = pd.DataFrame(params_list, columns=params.keys())
    return params_df


def test_generate_params_df(params: dict):
    """Test the generate_params_df function.

    Params:
        params (dict): The dictionary of parameters
    """
    params_df = generate_params_df(params)
    print(params_df)
    return params_df


# Test case
# params = {
#     'N': [4],
#     'W': [1],
#     'Q': [1],
#     'beta': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1]
# }
# params_df = test_generate_params_df(params)
# # for _, row in params_df.iterrows():
# # print(type(row))
# # print(type(params_df.iterrows()))