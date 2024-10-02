import itertools
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from utils.helper import get_date_time, get_fanatics

def visualize_beliefs_helper(beliefs: np.ndarray, ax: plt.Axes) -> plt.figure:
    """
    Visualize clusters of beliefs in different periods of the network. 
    This is the core function behind Graeme's Figure 1+4.

    People in the same subnetwork tend to have the same beliefs

    Params:
        beliefs (np.ndarray): The belief of each person in the network
        conv (int): the current period of the simulation. If 0, don't show. If -1, "converged".
        ax (plt.Axes): the figure's axes instance

    Returns:
        fig: The figure of the belief of people (in Graeme's paper)
    """
    # Sort belief and plot
    sorted_b = np.sort(beliefs)[::-1]

    # Plot figure
    ax.set_box_aspect(1/3)
    fig = ax.barh(list(range(len(sorted_b))), sorted_b, height=1)
    ax.xaxis.set_ticks(np.arange(0, 1.1, 0.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(sorted_b))
    ax.yaxis.set_visible(False)
    return fig

def visualize_network_helper(beliefs: np.ndarray, networks: np.ndarray, conv: int, ax: plt.Axes) -> plt.figure:
    """
    Visualize segments/clusters of the network.
    This is the core function behind Graeme's Figure 4.

    Params:
        beliefs (np.ndarray): The belief of each person in the network
        networks (np.ndarray): The network of people
        conv (int): The current period of the simulation
        ax (plt.Axes): the figure's axes instance

    Returns:
        fig: The figure of the networks of people
    """
    # Sort people according to beliefs
    members = np.argsort(np.argsort(beliefs))
    # Convert matrix to dictionary
    network = {i: np.where(networks[i] == 1) for i in range(networks.shape[0])}
    # Reindex people according to sorted beliefs
    sort_network = {members[i]: [members[fr]
                                 for fr in network[i]] for i in range(len(members))}

    # Create array for scatter plot
    sort_y = np.array([[y, y, y, y]
                      for y in list(sort_network.keys())]).flatten()
    sort_x = np.array(list(sort_network.values())).flatten()

    # Plot figure
    ax.set(xlim=(0, np.max(sort_x)), ylim=(np.max(sort_y), 0))
    ax.set_box_aspect(1)
    fig = ax.scatter(sort_x, sort_y, 0.2)
    conv = str(conv) if conv != -1 else "converged"
    ax.set_xlabel(f"t = {conv}")
    ax.tick_params(labelbottom='off')
    ax.yaxis.set_visible(False)

    return fig

def visualize_coevolution(data: pd.DataFrame, beta: float, date: str, time: str, export=False) -> plt.figure:
    """
    Graeme's Figure 4: Coevolution of Beliefs and Networks

    Params:
        results (pd.DataFrame): The results of the simulations
        fanatics_scheme (str): Fanatics scheme (max, min-max, mean)
        beta (float): Social dissonance value
        export (bool): Export this figure if `true`
        name (str): the OUTPUT_NAME of this program

    Returns:
        fig: The figure of networks & beliefs coevolution (in Graeme's paper)
    """
    date_time = get_date_time(date, time)
    FANATICS_COUNT = data['num_fanatics'].unique().size
    fig, axs = plt.subplots(nrows=4, ncols=FANATICS_COUNT*2, gridspec_kw={'width_ratios': [1, 3] * FANATICS_COUNT}, figsize=(FANATICS_COUNT * 4, 8))

    # Loop through different values of num_fanatics
    for i, num_fanatics in enumerate(data['num_fanatics'].unique()):
        df = data[(data['num_fanatics'] == num_fanatics)
                     & (data['beta'] == beta)]

        # Visualize belief at 4 different time steps
        timesteps = [1, 5, 20, -1]

        for j, time in enumerate(timesteps):
            visualize_network_helper(df['belief_array'].iloc[0][time],
                                     df['network_array'].iloc[0][time],
                                     time, axs[j, (i % FANATICS_COUNT)*2])
            # df['belief_array] is a list of np.ndarray of beliefs at time t
            visualize_beliefs_helper(df['belief_array'].iloc[0][time],
                                     axs[j, (i % FANATICS_COUNT)*2+1])

        axs[i // FANATICS_COUNT, (i % FANATICS_COUNT)*2].set_title(f"Fanatics = {num_fanatics}")
    
    fanatics_scheme = data["fanatics_scheme"].sample().str.cat(sep='')
    switching_prob = data["switching_prob"].sample().values[0]
    scheme = ''
    if fanatics_scheme == 'max':
        scheme = 'All fanatics hold the highest beliefs'
    elif fanatics_scheme == 'min-max':
        scheme = 'Half of the fanatics hold the highest beliefs, the other half hold the lowest beliefs'
    elif fanatics_scheme == 'mean':
        scheme = 'All fanatics hold the beliefs in the middle'
        
    fig.suptitle(f"Coevolution of beliefs and networks \n ({scheme}) \n (beta = {round(beta, 2)}, switching_prob = {switching_prob})")
    plt.tight_layout()

    # Save figure
    if export:
        try:
            os.mkdir(f'./plots/{date}')
        except FileExistsError:
            pass
        plt.savefig(f'./plots/{date}/{date_time}_coev_{fanatics_scheme}_{switching_prob}_{round(beta, 2)}.png', facecolor='white')
    plt.close()
    return fig

def visualize_clustering_coeff(data_fanatics: pd.DataFrame, data_no_fanatics: pd.DataFrame,
                               fanatics_scheme: list, beta: list, num_fanatics: int, switching_prob: float, 
                               date: str, time: str, export=False) -> plt.figure:
    # Line plot
    fig = plt.figure(figsize=(10,6))

    # Plot individual line for each param set
    for (fanatic_value, beta_value) in itertools.product(fanatics_scheme, beta):
        data = get_fanatics(data_fanatics, data_no_fanatics, fanatic_value, switching_prob = switching_prob)
        net_arr = data[(data['beta'] == beta_value) & (data['num_fanatics'] == num_fanatics)]["network_array"].reset_index(drop=True)[0]
        clustering_coeffs = np.zeros(shape=len(net_arr))

        # Calculate clustering coef
        for i in range(len(net_arr)):
            G = nx.from_numpy_array(net_arr[i])
            clustering_coeffs[i] = nx.average_clustering(G)
        
        plt.plot(np.arange(0, len(clustering_coeffs)), clustering_coeffs, label=f"{fanatic_value} {beta_value} {switching_prob}")
        
    # Customization
    plt.legend(loc="upper left")
    plt.title(f"Clustering coefficient of {num_fanatics} {fanatics_scheme} fanatics, beta {beta}, and switching prob {switching_prob}")
    plt.ylim([0, 0.45])

    # Export
    date_time = get_date_time(date, time)
    if export:
        plt.savefig(f"./plots/{date}/{date_time}_clustering_coeff_{num_fanatics}_{fanatics_scheme}_{switching_prob}_{beta}.png")

    return fig

def visualize_indegree(data: pd.DataFrame, date: str, time: str, export=False):
    """Visualize the distribution of in-degrees."""
    scheme = ''
    fanatics_scheme = data["fanatics_scheme"].sample().str.cat(sep='')
    switching_prob = data["switching_prob"].sample().values[0]
    if fanatics_scheme == 'max':
        scheme = "all fanatics' beliefs=1"
    elif fanatics_scheme == 'min-max':
        scheme = "half of the fanatics' beliefs=1, the other half=0"
    elif fanatics_scheme == 'mean':
        scheme = "all fanatics' beliefs=0.5"

    temp = data.explode('last_indegree')
    palette = sns.color_palette("hls", 10)
    g_indegree = sns.displot(data=temp, x='last_indegree', hue='beta', col='num_fanatics', col_wrap=4, palette=palette, multiple='stack', legend=False, stat='probability')
    g_indegree.set_axis_labels('In-degree', 'Count')
    g_indegree.set_titles('Number of fanatics: {col_name}')
    g_indegree.figure.suptitle(f'Distribution of in-degrees ({scheme}, switching prob = {switching_prob})', y=1.05)
    plt.legend(title='beta', labels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.show(g_indegree)

    # Export
    date_time = get_date_time(date, time)
    if export:
        g_indegree.savefig(f'./plots/{date}/{date_time}_indegree_{fanatics_scheme}_{switching_prob}.png')
        
def line_plot(data, metrics, date: str, time: str, export=False) -> plt.figure:
    """
    Create a line plot of the metric over the beta values.
    """
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(20, 10))

    for i, metric in enumerate(metrics):
        g = sns.lineplot(data=data, x='beta', y=metric, hue='num_fanatics', palette='viridis', ax=axs[i//4][i%4])
        if metric in ['MEAN', 'STDEV', 'BIAS', 'ENLITE', 'CON', 'CLUSTERING_COEFFICIENT']:
            g.set_ylim(0, 1)
        if metric in ['MEDIAN_INDEGREE', 'MODE_INDEGREE']:
            g.set_ylim(0, 8)
        if metric in ['MAX_INDEGREE']:
            g.set_ylim(0, 17)
        axs[i//4][i%4].set_title(f'{metric} vs. beta')
        sns.move_legend(g, "upper right", title='# fanatics')

    # Customization
    plt.tight_layout()
    fanatics_scheme = data["fanatics_scheme"].sample().str.cat(sep='')
    switching_prob = data["switching_prob"].sample().values[0]
    fig.suptitle(f'{fanatics_scheme} fanatic w\' switching prob = {switching_prob}', fontsize=16, y=1.02)
    
    # Export
    date_time = get_date_time(date, time)
    if export:
        plt.savefig(f'plots/{date}/{date_time}_metrics_{fanatics_scheme}_{switching_prob}.png')

    plt.show()

    return fig

def line_plot_95_interval(data, metrics, date: str, time: str, export=False) -> plt.figure:
    """
    Create a line plot of the metric over the beta values.
    """
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(20, 10))
    num_fanatics = data['num_fanatics'].unique()
    color_palette = sns.color_palette("tab20", len(num_fanatics) * 2).as_hex()

    for i, metric in enumerate(metrics):
        # Visualize lines with confidence interval
        curr_plot = axs[i//4][i%4]
        
        for j, num_fanatic in enumerate(num_fanatics):
            data_1d = data[data['num_fanatics'] == num_fanatic]            
            curr_plot.plot(data_1d['beta'], data_1d[metric], color=color_palette[j*2], label=num_fanatic)
            curr_plot.fill_between(x=data_1d['beta'], y1=data_1d[f'{metric}_upper_bound'], y2=data_1d[f'{metric}_lower_bound'], color=color_palette[j*2+1], alpha=0.2)
            
        # Set view limit
        if metric in ['MEAN', 'STDEV', 'BIAS', 'ENLITE', 'CON', 'CLUSTERING_COEFFICIENT']:
            curr_plot.set_ylim(0, 1)
        if metric in ['MEDIAN_INDEGREE', 'MODE_INDEGREE']:
            curr_plot.set_ylim(0, 8)
        if metric in ['MAX_INDEGREE']:
            curr_plot.set_ylim(0, 17)
        curr_plot.set_title(f'{metric} vs. beta')
        curr_plot.legend(loc="upper right", title='# fanatics')

    # Customization
    plt.tight_layout()
    fanatics_scheme = data["fanatics_scheme"].sample().str.cat(sep='')
    switching_prob = data["switching_prob"].sample().values[0]
    fig.suptitle(f'{fanatics_scheme} fanatic w\' switching prob = {switching_prob}', fontsize=16, y=1.02)
    
    # Export
    date_time = get_date_time(date, time)
    if export:
        plt.savefig(f'plots/{date}/{date_time}_metrics_{fanatics_scheme}_{switching_prob}.png')

    plt.show()

    return fig

def toimg(fig):
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def visualize_graph_helper(network: np.ndarray, beliefs: np.ndarray, ax: plt.Axes) -> plt.figure:
    """ Visualize the graphical network of people using library networkx. 
    People within the same of range of beliefs are colored the same.

    Calculate the mean belief of clusters of people within the same range of beliefs and annotate it on the graph.

    Params:
        network (np.ndarray): The network of people
        beliefs (np.ndarray): The beliefs of each person in the network
        conv (int): The current period of the simulation
        export (bool): Export this figure if `true`
        name (str): the OUTPUT_NAME of this program

    Returns:
        fig: a figure of the network of people
    """
    G = nx.from_numpy_array(network, create_using=nx.DiGraph)
    pos = nx.spring_layout(G, seed = 42)
    fig = nx.draw(G, pos, node_size=100, node_color=beliefs, vmin=0, vmax=1,
            cmap=plt.cm.Reds, edge_color='gray', ax=ax)
    # nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    plt.close()
    return fig

def visualize_graph(network: np.ndarray, beliefs: np.ndarray, timestep: int,
                    f: str, s: str, sp: float, b: str, 
                    date: str, time: str, export=False) -> plt.figure:
    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(15, 15),
                                gridspec_kw={'height_ratios': [1, 3]})
    plt.suptitle(f"Network of {f} {s} fanatics, beta {b}, switching prob {sp} \n at converged step")
    visualize_beliefs_helper(beliefs[timestep], axs[0])
    visualize_graph_helper(network[timestep], beliefs[timestep], axs[1])
    fig.tight_layout()
    
    date_time = get_date_time(date, time)

    # Export
    if export:
        try:
            os.mkdir(f'./plots/{date}')
        except FileExistsError:
            pass
        fig.savefig(f'plots/{date}/{date_time}_network_{f}_{s}_{sp}_{b}.png')
    return fig

def animate_network(data: pd.DataFrame, beta: float, num_fanatics: int, date: str, time: str):
    net_arr = data[(data['beta'] == beta) & (data['num_fanatics'] == num_fanatics)]['network_array'].reset_index(drop=True)[0]
    b_arr = data[(data['beta'] == beta) & (data['num_fanatics'] == num_fanatics)]['belief_array'].reset_index(drop=True)[0]
    fanatics_scheme = data["fanatics_scheme"].sample().str.cat(sep='')
    switching_prob = data["switching_prob"].sample().values[0]
    img_arr = []

    for i in range(len(net_arr)):
        fig = visualize_graph(net_arr, b_arr, i, num_fanatics, fanatics_scheme, switching_prob, beta, date, time)
        img_arr.append(toimg(fig))
    
    height, width, _ = img_arr[-1].shape
    size = (width, height)
    date_time = get_date_time(date, time)
    out = cv2.VideoWriter(f'./plots/{date}/{date_time}_network_{num_fanatics}_{fanatics_scheme}_{switching_prob}_{beta}.avi',
                          cv2.VideoWriter_fourcc(*'DIVX'), 3, size)
    for i in range(len(img_arr)):
        out.write(img_arr[i])
    out.release()