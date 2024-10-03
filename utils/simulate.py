import math
import numpy as np
import pandas as pd
import multiprocessing
import concurrent.futures
import time
from utils.coevolve import coevolve
from utils.metrics import calculate_metrics

def get_z_score_from_confident_level(confident_level):
    lookup_table = {
        0.9: 1.645,
        0.95: 1.96,
        0.99: 2.575,
        0.995: 2.81,
        0.999: 3.29,
    }
    
    return lookup_table[confident_level]

def calculate_statistics(metrics, confident_level=0.95):
    z_score = get_z_score_from_confident_level(confident_level)
    means = metrics.mean(axis=0)
    std = metrics.std(axis=0)
    n = len(metrics)
    lower_bounds = means - (z_score * (std / math.sqrt(n)))
    upper_bounds = means + (z_score * (std / math.sqrt(n)))
    
    return means, lower_bounds, upper_bounds

def simulate_helper(row, num_simulations, metrics_lst, timesteps):
    """
    Helper method for simulate in which one set of parameters is chosen.

    Params:
        row (pd.DataFrame): the pre-defined set of parameters
        num_simulations (int): number of simulations
        metrics_lst (list): a list of metrics
    Returns:
        row.name (int): the index of the row
        metrics (pd.DataFrame): a DataFrame of the metrics after running num_simulations simulations for a set of parameters
        last_simulation (dict): a dictionary of the results of the last simulation
    """
    start_time = time.perf_counter()
    # Unpack the entries of the parameters for a row. The order is important -- this is the order of the columns in the params DataFrame
    P, N, Q, W, A, B, beta, network_threshold, switching_cost, switching_prob, belief_difference, num_fanatics, fanatics_scheme = row.values[
        :13]

    metrics = pd.DataFrame(columns=metrics_lst)

    last_simulation = {}

    for i in range(num_simulations):
        seed = i
        results = coevolve(P=P, N=N, Q=Q, W=W, beta=beta, A=A, B=B,
                           network_threshold=network_threshold, switching_cost=switching_cost, 
                           switching_prob=switching_prob, belief_difference=belief_difference,
                           num_fanatics=num_fanatics, fanatics_scheme=fanatics_scheme,
                           SEED=seed)

        metrics.loc[i] = calculate_metrics(
            init_b=results['init_b'], revised_b=results['revised_b'], revised_network=results['revised_network'],
            convergence_period=results['converged_time'])
        
        if i == num_simulations-1:
            last_simulation['network_array'] = results['network_array']
            last_simulation['belief_array'] = results['belief_array']
            last_simulation['converged_time']  = results['converged_time']

    finish_time = time.perf_counter()
    print(
        f'Simulated parameterization {row.name+1} in {round(finish_time-start_time, 2)} second(s)')
    return row.name, metrics, last_simulation


def simulate(params: pd.DataFrame, num_simulations: int, num_processors: int, timesteps: list):
    # Create a pandas DataFrame to store the simulation results for various parameterizations
    metrics_lst = ['MEAN', 'STDEV', 'BIAS', 'ENLITE', 'SECTS', 'CON', 'CLUSTERING_COEFFICIENT',
                   'MEAN_INDEGREE', 'MEDIAN_INDEGREE', 'MODE_INDEGREE', 'MIN_INDEGREE', 'MAX_INDEGREE',
                   'CONVERGENCE_PERIOD']

    for metric in metrics_lst:
        params[metric] = np.nan
        params[f'{metric}_lower_bound'] = np.nan
        params[f'{metric}_upper_bound'] = np.nan

    # Store the last simulation's networks and beliefs of each parameterization
    last_sim_networks = []
    last_sim_beliefs = []

    # Specify the number of CPUs to use
    if multiprocessing.cpu_count() < num_processors:
        num_cpus = multiprocessing.cpu_count()
    else:
        num_cpus = num_processors

    # Use a context manager to distribute the processes
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
        futures = []
        start_time = time.perf_counter()
        for _, row in params.iterrows():
            # Each parameterization corresponds to a `future` object
            future = executor.submit(
                simulate_helper, row, num_simulations, metrics_lst, timesteps)
            futures.append(future)
        for future in futures:
            # print(future)
            row, metrics, last_simulation = future.result()
            means, lower_bounds, upper_bounds = calculate_statistics(metrics, confident_level=0.95)
            for metric in metrics.columns:
                params.loc[row, metric] = means[metric]
                params.loc[row, f'{metric}_lower_bound'] = lower_bounds[metric]
                params.loc[row, f'{metric}_upper_bound'] = upper_bounds[metric]
                params.loc[row, 'converged_time']  = last_simulation['converged_time']
            last_sim_networks.append(last_simulation['network_array'])
            last_sim_beliefs.append(last_simulation['belief_array'])
    

        params['network_array'] = last_sim_networks
        params['belief_array'] = last_sim_beliefs
        finish_time = time.perf_counter()
        print(f'Finished in {round(finish_time-start_time, 2)} second(s)')

    return params
