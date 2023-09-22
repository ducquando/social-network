import numpy as np
from copy import copy
from utils.network import initialize_network, revise_network
from utils.belief import initialize_beliefs, revise_beliefs
import time
import networkx as nx


def coevolve(P: int, N: int, W: int, Q: int, A: float, B: float, beta: float, switching_cost: float, network_threshold: int, belief_difference: float, num_fanatics: int, fanatics_scheme: str, SEED: int):
    # start = time.perf_counter()
    np.random.seed(SEED)
    # Init
    init_network = initialize_network(P=P, N=N)
    init_b, fanatics = initialize_beliefs(
        P=P, Q=Q, A=A, B=B, num_fanatics=num_fanatics, fanatics_scheme=fanatics_scheme)
    results = {'init_network': init_network, 'init_b': init_b}
    revised_b = copy(init_b)
    revised_network = copy(init_network)

    belief_array = []
    belief_array.append(init_b)
    network_array = np.empty((0, P, P))
    network_array = np.concatenate(
        (network_array, np.expand_dims(init_network, axis=0)), axis=0)

    converged = False
    num_iter = 1
    while not converged:
        prev_b = revised_b
        prev_network = revised_network
        # Revise beliefs and network
        revised_b = revise_beliefs(network=prev_network, prev_b=prev_b, P=P,
                                   N=N, beta=beta, num_fanatics=num_fanatics, fanatics=fanatics)
        revised_network = revise_network(
            prev_network=prev_network, b=revised_b, W=W, switching_cost=switching_cost)

        # Store the revised network and beliefs to their corresponding arrays
        belief_array.append(revised_b)
        network_array = np.append(
            network_array, np.array([revised_network]), axis=0)

        # Convergence criterion: the network has not changed for the last network_threshold periods and the difference between the previous beliefs and the current beliefs is less than belief_difference
        # np.sum(np.abs(revised_b - prev_b)) < belief_difference
        if num_iter >= network_threshold and np.array_equal(network_array[num_iter], network_array[num_iter-network_threshold]) and np.sum([np.sum(np.abs(belief_array[i] - belief_array[i-1])) for i in range(num_iter-network_threshold, num_iter)]) < belief_difference:
            # print("Network converged at time step", num_iter)
            converged = True
            results['num_fanatics'] = num_fanatics
            results['converged_time'] = num_iter
            results['revised_b'] = revised_b
            results['revised_network'] = revised_network
            results['belief_array'] = belief_array
            results['network_array'] = network_array
            G = nx.from_numpy_array(revised_network, create_using=nx.DiGraph)
            break
        num_iter += 1
    # finish = time.perf_counter()
    # print(f'Finished in {round(finish-start, 2)} second(s)')
    # print(f'Each iteration took {round((finish-start)/num_iter, 2)} second(s) on average')
    return results
