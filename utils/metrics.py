import numpy as np
import networkx as nx
from utils.belief import calculate_enlightenment


def calculate_metrics(init_b: np.ndarray, revised_b: np.ndarray, revised_network: np.ndarray,
                      convergence_period):
    """
    Calculate the metrics of the simulation results.
    """
    P = len(init_b)
    # Calculate metrics
    # Belief-related metrics
    mean = np.mean(revised_b)
    stdev = np.std(revised_b)
    bias = np.abs(np.mean(revised_b)-np.mean(init_b))
    enlite = calculate_enlightenment(init_b=init_b, final_b=revised_b)

    # Network-related metrics
    G = nx.from_numpy_array(revised_network, create_using=nx.DiGraph)
    num_sects = nx.number_weakly_connected_components(G)
    sects_lst = [list(c) for c in sorted(
        nx.weakly_connected_components(G), key=len, reverse=True)]
    sects_size = [len(sect) for sect in sects_lst]
    con = sum(sect_size * (sect_size - 1) / (P * (P - 1))
              for sect_size in sects_size)
    clustering_coefficient = nx.average_clustering(G)

    # Indegree-related metrics
    indegree_lst = [G.in_degree(n) for n in G.nodes()]
    mean_indegree = np.mean(indegree_lst)
    median_indegree = np.median(indegree_lst)
    mode_indegree = np.argmax(np.bincount(indegree_lst))
    min_indegree = np.min(indegree_lst)
    max_indegree = np.max(indegree_lst)

    return [mean, stdev, bias, enlite, num_sects, con, clustering_coefficient, mean_indegree, median_indegree, mode_indegree, min_indegree, max_indegree, convergence_period]
