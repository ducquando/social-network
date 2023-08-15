from random import sample
import numpy as np


def initialize_network(P: int, N: int) -> np.ndarray:
    """ Create a 2D NumPy array representing a network of people. 

    Each person i has num_friends friends.
    For entry (i, j), if it is 0, it means that person i and j are not friends. If it is 1, it means that person i and j are friends. The network is not symmetric. For example, if (i, j) is 1, it does not mean that (j, i) is 1. 

    Params:
        P (int): The number of people in the network (population size)
        N (int): The number of friends each person has (network size)

    Returns:
        a 2D NumPy array representing the network
    """
    network = np.zeros((P, P))
    for i in range(P):
        strangers = list(range(P))
        strangers.remove(i)
        friends = sample(strangers, N)
        for j in friends:
            network[i][j] = 1
    return network


def revise_network_old(prev_network: np.ndarray, b: np.ndarray, W: int, switching_cost: float) -> np.ndarray:
    """Revise the network of people.

    After every period t and after beliefs are revised, every person i revise her network by randomly meeting W new potential associates, who at the time are not their friends yet, and accessing their beliefs. If the belief of a potential associate j (person j is currently not in person i's network) is closer to person i's belief than the belief of one of the friends who is furthermost from her own, then person i will replace this furthermost friend by this new potential associate to her network.

    Params:
        prev_network (np.ndarray): The previous network of people
        W (int): The number of new potential associates to meet
        b (np.ndarray): The beliefs of each person in the network

    Returns:
        nw_revised: a 2D NumPy array representing the revised network of people
    """
    nw_revised = prev_network.copy()
    for i in range(prev_network.shape[0]):
        strangers = np.where(prev_network[i] == 0)[0]
        if len(strangers) > 0:
            # Randomly select W new potential associates from the strangers
            # If the number of strangers is less than W, then select all strangers
            new_associates = sample(list(strangers), k=min(W, len(strangers)))
            friends = np.where(prev_network[i] == 1)[0]
            furthest_friend = friends[np.argmax(np.abs(b[friends] - b[i]))]
            furthest_dissonance = np.abs(b[furthest_friend] - b[i])
            if len(new_associates) > 0:
                for j in new_associates:
                    # Since the entry (i,i) is 0, we need to make sure that the person i herself in not in the list of strangers
                    if j != i:
                        new_dissonance = np.abs(b[j] - b[i])
                        if furthest_dissonance - new_dissonance > switching_cost:
                            nw_revised[i][furthest_friend] = 0
                            nw_revised[i][j] = 1
                    # Update the friendlist of person i regardless of whether she decides to swap j for her worst friend or not
                    # In the next iteration of j, this "newly swapped" associate will be considered one of her friends
                    friends = np.where(nw_revised[i] == 1)[0]
    return nw_revised


def revise_network(prev_network: np.ndarray, b: np.ndarray, W: int, switching_cost: float) -> np.ndarray:
    """
    Revise the network of people.

    After every period t and after beliefs are revised, every person i revise her network by randomly meeting W new potential associates, who at the time are not their friends yet, and accessing their beliefs. If the belief of a potential associate j (person j is currently not in person i's network) is closer to person i's belief than the belief of one of the friends who is furthermost from her own, then person i will replace this furthermost friend by this new potential associate to her network.

    Params:
        prev_network (np.ndarray): The previous network of people.
        W (int): The number of new potential associates to meet.
        b (np.ndarray): The beliefs of each person in the network.
        switching_cost (float): The threshold for swapping friends.

    Returns:
        revised_network: a 2D NumPy array representing the revised network of people.
    """
    revised_network = prev_network.copy()
    P = prev_network.shape[0]
    for i in range(P):
        # np.where returns a tuple, of which the first element is the array of indices; convert the returned numpy array to a list to remove person i
        strangers = list(np.where(prev_network[i] == 0)[0])
        strangers.remove(i)
        friends = np.where(prev_network[i] == 1)[0]
        furthest_friend = friends[np.argmax(np.abs(b[friends] - b[i]))]
        furthest_dissonance = np.abs(b[furthest_friend] - b[i])
        if len(strangers) > 0:
            new_associates = sample(strangers, k=min(W, len(strangers)))
            chosen_associates = [
                j for j in new_associates if furthest_dissonance - np.abs(b[j] - b[i]) > switching_cost]
            if len(chosen_associates) > 0:
                closest_associate = chosen_associates[np.argmin(
                    np.abs(b[chosen_associates] - b[i]))]
                revised_network[i][furthest_friend] = 0
                revised_network[i][closest_associate] = 1
    return revised_network

def test_revise_network():
    np.random.seed(42)
    init_network = initialize_network(10, 2)
    b = np.random.beta(a=0.005, b=0.005, size=10)
    revised_network = revise_network(init_network, b, 1, 0.1)
    print(init_network)
    print(revised_network)

# test_revise_network()