# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import networkx as nx
# from tqdm import tqdm


# def visualize_beliefs_helper(beliefs: np.ndarray, conv: int, ax: plt.Axes) -> plt.figure:
#     """
#     Visualize clusters of beliefs in different periods of the network. 
#     This is the core function behind Graeme's Figure 1+4.

#     People in the same subnetwork tend to have the same beliefs

#     Params:
#         beliefs (np.ndarray): The belief of each person in the network
#         conv (int): the current period of the simulation. If 0, don't show. If -1, "converged".
#         ax (plt.Axes): the figure's axes instance

#     Returns:
#         fig: The figure of the belief of people (in Graeme's paper)
#     """
#     # Sort belief and plot
#     sorted_b = np.sort(beliefs)[::-1]
#     ax.set_box_aspect(1/3)
#     fig = ax.barh(list(range(len(sorted_b))), sorted_b, height=1)

#     # Change conv to string if conv != 0
#     if conv:
#         conv = str(conv) if conv != -1 else "converged"
#         ax.set_xlabel(f"t = {conv}")
#     ax.xaxis.set_ticks([0, 0.5, 1])
#     ax.yaxis.set_visible(False)

#     return fig


# def visualize_network_helper(beliefs: np.ndarray, networks: np.ndarray, conv: int, ax: plt.Axes) -> plt.figure:
#     """
#     Visualize segments/clusters of the network.
#     This is the core function behind Graeme's Figure 4.

#     Params:
#         beliefs (np.ndarray): The belief of each person in the network
#         networks (np.ndarray): The network of people
#         conv (int): The current period of the simulation
#         ax (plt.Axes): the figure's axes instance

#     Returns:
#         fig: The figure of the networks of people
#     """
#     # Sort people according to beliefs
#     members = np.argsort(np.argsort(beliefs))
#     # Convert matrix to dictionary
#     network = {i: np.where(networks[i] == 1) for i in range(networks.shape[0])}
#     # Reindex people according to sorted beliefs
#     sort_network = {members[i]: [members[fr]
#                                  for fr in network[i]] for i in range(len(members))}

#     # Create array for scatter plot
#     sort_y = np.array([[y, y, y, y]
#                       for y in list(sort_network.keys())]).flatten()
#     sort_x = np.array(list(sort_network.values())).flatten()

#     # Plot figure
#     ax.set(xlim=(0, np.max(sort_x)), ylim=(np.max(sort_y), 0))
#     ax.set_box_aspect(1)
#     fig = ax.scatter(sort_x, sort_y, 0.2)
#     if conv:
#         conv = str(conv) if conv != -1 else "converged"
#         ax.set_xlabel(f"t = {conv}")
#         ax.tick_params(labelbottom='off')
#     ax.yaxis.set_visible(False)

#     return fig


# def visualize_beliefs_coev(results: pd.DataFrame, beta: float, export=False, name="") -> plt.figure:
#     """
#     Graeme's Figure 1: Evolution of Individual Beliefs without Network Changes

#     Params:
#         results (pd.DataFrame): The results of the simulations
#         beta (float): Social dissonance value
#         export (bool): Export this figure if `true`
#         name (str): the OUTPUT_NAME of this program

#     Returns:
#         fig: The figure of the belief of people (in Graeme's paper)
#     """
#     # 10 different graphs for 10 fanatic values
#     fig, axs = plt.subplots(nrows=4, ncols=8, figsize=(15, 12))

#     # Loop through different values of num_fanatics
#     for i, num_fanatics in enumerate(results['num_fanatics'].unique()):
#         # Visualize belief at 4 different time steps
#         timesteps = [0, 5, 15, -1]
#         for j, time in enumerate(timesteps):

#             df = results[(results['num_fanatics'] == num_fanatics)
#                          & (results['beta'] == beta)]

#             visualize_beliefs_helper(
#                 df['belief_array'].iloc[0][time], time, axs[j+(i//5)*5, i % 5])

#         # Add blank space between rows
#         axs[4, i % 5].set_visible(False)
#         axs[(i//5)*5, i % 5].set_title(f"Fanatics = {num_fanatics}")

#     fig.suptitle(f"Evolution of individual beliefs \n where beta = {beta}")
#     plt.tight_layout()

#     # Save figure
#     if export:
#         plt.savefig(f"./plots/{name}_Graeme's belief (beta = {beta}).png")
#     plt.close()
#     return fig


# def visualize_network_coev(results: pd.DataFrame, beta: float, export=False, name="") -> plt.figure:
#     """
#     Graeme's Figure 4: Coevolution of Beliefs and Networks

#     Params:
#         results (pd.DataFrame): The results of the simulations
#         beta (float): Social dissonance value
#         export (bool): Export this figure if `true`
#         name (str): the OUTPUT_NAME of this program

#     Returns:
#         fig: The figure of networks & beliefs coevolution (in Graeme's paper)
#     """
#     # 10 different graphs for 10 fanatic values
#     fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(25, 15),
#                             gridspec_kw={'width_ratios': [1, 3, 1, 3, 1, 3, 1, 3]})

#     # Loop through different values of num_fanatics
#     for i, num_fanatics in enumerate(results['num_fanatics'].unique()):
#         # Visualize belief at 4 different time steps
#         df = results[(results['num_fanatics'] == num_fanatics)
#                      & (results['beta'] == beta)]
#         timesteps = [0, 5, 15, -1]
#         for j, time in enumerate(timesteps):
#             visualize_network_helper(df['belief_array'].iloc[0, time],
#                                      df['network_array'].iloc[0, time],
#                                      time, axs[j, (i % 4)*2])
#             visualize_beliefs_helper(df['belief_array'].iloc[0, time],
#                                      0, axs[j, (i % 4)*2+1])
#         # Add blank space between rows
#         # axs[4, (i % 4)*2].set_visible(False)
#         # axs[4, (i % 4)*2+1].set_visible(False)
#         # axs[(i//4)*4, (i % 4)*2].set_title(f"Fanatics = {num_fanatics}")
#     fig.suptitle(f"Coevolution of beliefs and networks \n where beta = {beta}")
#     plt.tight_layout()

#     # Save figure
#     if export:
#         plt.savefig(f"./plots/{name}_Graeme's network (beta = {beta}).png")
#     plt.close()
#     return fig

# # ------------------------------------------------------------------------------
# # Modern built-in methods for better visualization


# def visualize_network(network: np.ndarray, beliefs: np.ndarray, conv: int, export=False, name="") -> plt.figure:
#     """ Visualize the network of people using library networkx. 
#     People within the same of range of beliefs are colored the same.

#     Calculate the mean belief of clusters of people within the same range of beliefs and annotate it on the graph.

#     Params:
#         network (np.ndarray): The network of people
#         beliefs (np.ndarray): The beliefs of each person in the network
#         conv (int): The current period of the simulation
#         export (bool): Export this figure if `true`
#         name (str): the OUTPUT_NAME of this program

#     Returns:
#         fig: a figure of the network of people
#     """
#     G = nx.from_numpy_array(network, create_using=nx.DiGraph)
#     pos = nx.spring_layout(G)
#     fig, ax = plt.subplots(figsize=(10, 10))
#     nx.draw(G, pos, node_size=100, node_color=beliefs,
#             cmap=plt.cm.RdYlBu, ax=ax)
#     nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
#     plt.title(f"Network at time step {conv}")
#     fig.tight_layout()

#     # Save figure
#     if export:
#         plt.savefig(f"./plots/{name}_network (t={conv}).png")
#     plt.close()
#     return fig


# def visualize_beliefs(init_b: np.ndarray, revised_b: np.ndarray, conv: int, export=False, name="") -> plt.figure:
#     """
#     Visualize beliefs distribition in the network in several timesteps.

#     Params:
#         beliefs (np.ndarray): The beliefs of each person in the network
#         conv (int): The current period of the simulation
#         export (bool): Export this figure if `true`
#         name (str): the OUTPUT_NAME of this program

#     Returns:
#         fig: a Matplotlib figure object visualizing the belief distribution
#     """
#     fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
#     axs[0, 0].hist(init_b, bins=20, range=(0, 1))
#     axs[0, 1].hist(init_b, bins=20)
#     axs[0, 0].set_title("Initial beliefs, x = (0,1)")
#     axs[0, 1].set_title("Initial beliefs")
#     axs[1, 0].hist(revised_b, bins=20, range=(0, 1))
#     axs[1, 1].hist(revised_b, bins=20)
#     axs[1, 0].set_title("Final beliefs, x = (0,1)")
#     axs[1, 1].set_title("Final beliefs")
#     fig.suptitle(f"Belief distribution after {conv} time steps")
#     plt.tight_layout()

#     # Save figure
#     if export:
#         plt.savefig(f"./plots/{name}_Belief (t={conv}).png")
#     plt.close()
#     return fig


# def visualize_metrics(results: pd.DataFrame, metric: str, export=False, name="") -> plt.figure:
#     """
#     Visualize the correlation between beta (x-axis) and other metrics (y-axis), grouped by num_fanatics.

#     Params:
#         results (pd.DataFrame): The results of the simulations
#         metric (str): The metric to plot, which are BIAS, SECTS, MEAN, CON, STDEV, and ENLITE
#         conv (int): The current period of the simulation
#         export (bool): Export this figure if `true`
#         name (str): the OUTPUT_NAME of this program

#     Returns:
#         fig: a Matplotlib figure object visualizing the correlation
#     """
#     # Change nrows and ncols based on the number of different values of the number of fanatics.
#     # Here we have 11 different values of num_fanatics.
#     fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(10, 7))

#     if metric != "SECTS":
#         y_axis = np.linspace(0, 1, 6)
#     else:
#         y_axis = np.arange(0, 11, 2)

#     # Loop through different values of num_fanatics
#     for i, num_fanatics in enumerate(results['num_fanatics'].unique()):
#         df = results[(results['num_fanatics'] == num_fanatics)]
#         df.plot(x='beta', y=metric,
#                 ax=axes[i // 4, i % 4], sharey=True, yticks=y_axis)
#         # Visualize standard error for each value of beta
#         # df.groupby('beta').agg({metric: ['MEAN', 'STD']}).plot(yerr=metric, ax=axes[i//5, i%5])
#         axes[i//4, i % 4].set_title(f'Number of fanatics = {num_fanatics}')
#         axes[i//4, i % 4].set_ylabel(metric)
#     fig.suptitle(f"Fanatics and {metric}")
#     fig.tight_layout()

#     # Save figure
#     if export:
#         fig.savefig(f'./plots/{name}_{metric}.png')
#     plt.close()
#     return fig


# def visualize_indegree_helper(network: np.ndarray, fanatics: int, beta: float, ax: plt.Axes) -> plt.figure:
#     """
#     Visualize the indegree distribution

#     Params: 
#         network (np.ndarray): The network of people
#         fanatics (int) & beta (float): input parameterizations
#         export (bool): Export this figure if `true`
#         name (str): the OUTPUT_NAME of this program

#     Returns:
#         fig: a Matplotlib figure object visualizing the network's indegree centrality
#     """
#     # Calculate in-degree centrality
#     G = nx.from_numpy_array(network, create_using=nx.DiGraph)
#     degrees = [G.in_degree(n) for n in G.nodes()]

#     # Plot figure
#     fig = ax.hist(degrees, bins=np.arange(0, 101, 4))
#     ax.set_yticks(np.arange(0, 151, 40))
#     ax.set_title(f"fanatics = {fanatics}, beta = {beta}")
#     plt.close
#     return fig


# def visualize_indegree(results: pd.DataFrame, conv: float, export=False, name="") -> plt.figure:
#     """
#     Visualize the indegree distribution

#     Params: 
#         results (pd.DataFrame): The results of the simulations
#         conv (int): The current period of the simulation
#         export (bool): Export this figure if `true`
#         name (str): the OUTPUT_NAME of this program

#     Returns:
#         fig: a Matplotlib figure object visualizing the network's indegree centrality
#     """
#     # 100 different graphs for 10 num_fanatics & 10 beta values
#     fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(40, 50))

#     # Loop through different values of num_fanatics and beta
#     for i, num_fanatics in enumerate(results['num_fanatics'].unique()):
#         for j, beta in enumerate(results['beta'].unique()):
#             df = results[(results['num_fanatics'] == num_fanatics)
#                          & (results['beta'] == beta)]
#             visualize_indegree_helper(
#                 df['network_array'].iloc[0][conv], num_fanatics, beta, axs[j % 10, i % 10])
#     conv = str(conv) if conv != -1 else "converged"
#     fig.suptitle(f'Indegree distribution \n where time = {conv}')
#     plt.tight_layout()

#     # Save figure
#     if export:
#         fig.savefig(f'./plots/{name}_Indegree distribution (time={conv}).png')
#     plt.close()
#     return fig


# def visualize_eigenvector(network: np.ndarray, eigenvector: int, df: pd.DataFrame, conv: int, export=False, name="") -> plt.figure:
#     """
#     Visualize the network in term of eigenvector centrality 

#     Params: 
#         network (np.ndarray): The network of people
#         eigenvector (int): The number of eigenvector
#         df (pd.DataFrame): A DataFrame to store nodes' index
#         conv (int): The current period of the simulation
#         export (bool): Export this figure if `true`
#         name (str): the OUTPUT_NAME of this program

#     Returns:
#         fig: a Matplotlib figure object visualizing the network's eigenvector centrality
#     """
#     G = nx.from_numpy_array(network, create_using=nx.DiGraph)
#     fig, ax = plt.subplots(figsize=(10, 10))

#     # Calculate eigenvector centrality using built-in function with max iteration = 1000, and tolerance = 1.0e-3
#     eigen_centrality = nx.eigenvector_centrality(G, max_iter=10000, tol=1.0e-3)
#     all_nodes = [(node, eigen_centrality[node]) for node in eigen_centrality]
#     eigen_array = np.array(list(eigen_centrality.values()))

#     # Sort the eigenvector array and take the nodes with high eigenvector centrality only
#     top_nodes = [n for (n, c) in all_nodes if c in np.sort(
#         eigen_array)[-eigenvector:]]

#     # Create a subgraph with selected nodes
#     G1 = G.subgraph(top_nodes)
#     # Calculate the eigenvector centrality for the subgraph
#     top_centrality = nx.eigenvector_centrality(G1, max_iter=10000, tol=1.0e-3)

#     # Draw network
#     df[conv] = np.array(list(top_centrality.keys()))
#     nx.draw_spring(G1, node_color=G1.nodes(), node_size=[
#                    top_centrality[n] for n in G1.nodes()], font_size=10, with_labels=True)
#     plt.title(
#         label=f"Network with eigenvector centrality at time {conv} \n Top nodes are : {top_centrality.keys()}")
#     fig.tight_layout()

#     # Save figure
#     if export:
#         fig.savefig(f'./plots/{name}_Eigenvector (t={conv}).png')
#     plt.close()
#     return fig

# # ------------------------------------------------------------------------------
# # Main function


# def create_visuals(df, betas: list, name: str):
#     """
#     Create all visuals for this project

#     Params:
#         df (pd.DataFrame): list of sets of parameters
#         betas (list): list of all social dissonance values
#         name (str): this project's name
#     """
#     print("Creating visuals...")
#     with tqdm(total=20) as progress:
#         for metric in ['MEAN', 'STDEV', 'ENLITE', 'BIAS', 'CON', 'SECTS']:
#             visualize_metrics(df, metric, export=True, name=name)
#             progress.update()

#         # Visualize the last coevolution
#         for beta in betas:
#             visualize_beliefs_coev(df, beta, export=True, name=name)
#             visualize_network_coev(df, beta, export=True, name=name)
#             progress.update()

#         for time in [0, -1]:
#             visualize_indegree(df, time, export=True, name=name)
#             progress.update()
