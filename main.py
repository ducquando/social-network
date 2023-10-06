"""
Script to run simulations and save data in data files.

Terminal input: python main.py [num simulations] [num_processors]
Example: python main.py 100 8 (for 100 simulations per parameterization with 8 processors and default output file name)
"""
import sys
import os
import numpy as np
import time
from utils.params import generate_params_df
from utils.simulate import simulate


def main():
    NUM_SIMULATIONS = int(sys.argv[1])
    NUM_PROCESSORS = int(sys.argv[2])
    TIMESTEPS = [1, 5, 25, -1]

    params = {
        'P': [200],     # Population size
        # 'P': [5, 8],  # for testing purposes
        'N': [4],       # Personal network size
        'Q': [1],       # Number of random draws
        'W': [1],       # Number of new associates to meet each period
        'A': [.005],    # Define shape of beta distribution
        # 'A': [0.003],   # for replicating purpose (same as in the paper)
        'B': [.005],    # "",
        # 'B': [0.002],   # for replicating purpose (same as in the paper)
        # Social dissonance
        'beta': np.linspace(0.1, 1., 10),
        # Period difference to check convergence
        'network_threshold': [20],
        'switching_cost': [.000001],    # Threshold for swapping friends
        'belief_difference': [.001],    # Threshold for convergence
        'num_fanatics': None,            # Number of fanatics in population
        'fanatics_scheme': None,    # Scheme for choosing fanatics
    }

    params_no_fanatics = params.copy()
    params_no_fanatics['num_fanatics'] = [0]
    params_no_fanatics['fanatics_scheme'] = ['none']

    params_fanatics = params.copy()
    params_fanatics['num_fanatics'] = np.array([20, 25, 30, 40, 50, 60, 75, 80, 90, 100])
    params_fanatics['fanatics_scheme'] = ['max', 'min-max', 'mean']

    params_no_fanatics_df = generate_params_df(params_no_fanatics)
    params_fanatics_df = generate_params_df(params_fanatics)

    t = time.localtime()

    OUTPUT_NAME = f'{time.strftime("%y%m%d", t)}_{time.strftime("%H%M", t)}'

    # Store networks and beliefs with all other metrics
    simulations_results_no_fanatics = simulate(
        params_no_fanatics_df, NUM_SIMULATIONS, NUM_PROCESSORS, TIMESTEPS)
    simulations_results_fanatics = simulate(
        params_fanatics_df, NUM_SIMULATIONS, NUM_PROCESSORS, TIMESTEPS)

    # Create a folder for the date if it does not exist
    try:
        os.mkdir(f'./data/{time.strftime("%y%m%d", t)}')
    except FileExistsError:
        pass

    # Store simulation results in a pickle file (so that numpy arrays are not read as `str`)
    simulations_results_no_fanatics.to_pickle(
        f'./data/{time.strftime("%y%m%d", t)}/{OUTPUT_NAME}_no_fanatics.pkl')
    simulations_results_fanatics.to_pickle(
        f'./data/{time.strftime("%y%m%d", t)}/{OUTPUT_NAME}_fanatics.pkl')


if __name__ == '__main__':
    main()
