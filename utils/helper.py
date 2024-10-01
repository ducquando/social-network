import networkx as nx
import numpy as np
import pandas as pd

def get_date_time(date: str, time: str) -> str:
    return date + '_' + time

def get_indegree(network):
    G = nx.from_numpy_array(network, create_using=nx.DiGraph)
    indegree = np.array([G.in_degree(n) for n in G.nodes()])
    return indegree

def get_fanatics(data_fanatics: np.ndarray, data_no_fanatics: np.ndarray, scheme: str, switching_prob: float):
    """
    Concatenate the results of the simulations with fanatics and the results of the simulations without fanatics for a fanatic scheme

    Params:
        data_fanatics (np.ndarray): Fanatics datasets
        data_no_fanatics (np.ndarray): Non-fanatics dataset
        fanatics_scheme (str): Fanatics scheme (max, min-max, mean)
    """
    data = data_fanatics[data_fanatics['fanatics_scheme'] == scheme]
    data = pd.concat([data_no_fanatics, data])
    data = data[data['switching_prob'] == switching_prob]
    data['fanatics_scheme'] = scheme

    return data

def get_data(date: str, time: str):
    path = './data/' + date + '/'
    date_time = get_date_time(date, time)

    path_fanatics = path + date_time + '_fanatics.pkl'
    data_fanatics = pd.read_pickle(path_fanatics)

    path_no_fanatics = path + date_time + '_no_fanatics.pkl'
    data_no_fanatics = pd.read_pickle(path_no_fanatics)
    
    return data_fanatics, data_no_fanatics