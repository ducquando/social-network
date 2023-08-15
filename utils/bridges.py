import numpy as np
import networkx as nx
from functools import reduce

def sort_betweenness(network: np.ndarray, **kwargs) -> dict:
    """ Calculate the normalized betweenness centrality for all nodes in time t.
    
    We normalize each node's betweenness centrality by dividing it for the total number of possible edges in their network component
    
    Params:
        network (np.ndarray): The network of people
        
    Returns:
        top_10_between (dict): a dictionary of top 10 nodes of highest normalized betweenness centrality
    """
    SEED = kwargs.get("seed", 295)
    network = nx.from_numpy_array(network) 
    
    # Calculate the betweenness centrality
    all_betweeness = nx.betweenness_centrality(network, normalized = False, 
                                               weight = None, seed = SEED)
    
    # Normalize the betweenness centrality
    for each in all_betweeness.keys():
        component_size = len(nx.node_connected_component(network, each)) 
        all_betweeness[each] = all_betweeness[each] / ((component_size - 1) * (component_size - 2))

    # Sort the dictionary based on normalized betweenness centralities in DESC order
    top_between = sorted(all_betweeness.items(), key = lambda item: item[1], reverse = True)
    
    # Take the top 10 nodes and convert it into a dictionary
    top_10_between = dict(top_between[0:10])
    
    return top_10_between

def mean_belief(nodes: list, BELIEFS: np.ndarray) -> float:
    """ Average the belief of all people in the given group
    
    Params:
        nodes (list): a group of people
        BELIEFS (np.ndarray): the beliefs of all people in a network
        
    Returns:
        mean (float): the mean belief of those people
    """

    # Compute the sum of all nodes' belief
    sum_belief = reduce(lambda x, y: x + y, [BELIEFS[node] for node in nodes])
    
    # Average them
    mean = sum_belief / len(nodes)
    
    return mean

def has_diff_mean_belief(groups: list, BELIEFS: np.ndarray) -> bool:
    """ Compare different groups of people in terms of their mean bielfs
    
    Params:
        groups (list): The list of groups of people 
        BELIEFS (np.ndarray): the beliefs of all people in a network
    
    Returns:
        TRUE if the difference in terms of mean belief is larger than the pre-specified threshold
        FALSE otherwise
    
    """
    # Specify the difference threshold
    THRESHOLD = 0.0001
    
    # Calculate the mean belief of each group
    mean_beliefs = []
    for group in groups:
        mean_beliefs.append(mean_belief(group, BELIEFS))
    
    # Return whether or not the difference in belief is larger than the threshold
    return reduce(lambda a, b: a or b, [abs(x - y) > THRESHOLD for x in mean_beliefs for y in mean_beliefs if x != y])

def bridges(network: np.ndarray, BELIEFS) -> list:
    """ Return all the bridges consisting in the network in time t
    
    The bridges are defined as the NODES whose removal will divide its current component into at least two subcomponents of different beliefs
    
    Params:
        network (np.ndarray): The network of people in time t
        BELIEFS (np.ndarray): the beliefs of all people in a network in time t
        
    Returns:
        all_bridges (list): a list containing all bridges in the network
    """
    # Convert the network into undirected network
    network = nx.from_numpy_array(network)
    
    # Return the list of all possible articulation points
    all_bridges = list(nx.articulation_points(network))
    
    # Check if they connect different-belief subcomponents
    lst_bridges = []
    for bridge in all_bridges:
        new_network = network.copy()
        new_network.remove_node(bridge)
        
        # List all the newly disconnected components resulting from our node removal
        seperated_components = [c for c in list(nx.connected_components(new_network)) if c not in list(nx.connected_components(network))]
        
        # If the node removal creates difference in belief, that node is a bridge
        if has_diff_mean_belief(seperated_components, BELIEFS):
            lst_bridges.append(bridge)
    
    return lst_bridges