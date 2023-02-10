import random
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt


def set_random_seed(seed):
    print("Setting random seed: ", seed)
    random.seed(seed)
    torch.cuda.cudnn_enabled = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False


def plot_scenario(scenario_table, name=None):
    """
    :param scenario_table: a C x E tensor (C: # of classes, E: # of experiences)
    :return:
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))


    # Colors
    n_classes = scenario_table.shape[0]
    n_e = scenario_table.shape[1]
    cmap = sns.color_palette("deep", n_classes+1)
    cmap[0] = (1, 1, 1)

    # Heat Map
    table_plot = scenario_table.clone()
    for i in range(table_plot.shape[0]):
        idx = table_plot[i] > 0.0
        table_plot[i][idx] = i + 1
    sns.heatmap(table_plot, cbar=False, cmap=cmap, vmin=0, ax=ax,
                xticklabels=[str(i) if i % 2 ==0 else '' for i in range(n_e) ],
                yticklabels=False, linewidths=0.5)

    ax.set_xlabel("Experience", fontsize=12)
    ax.set_ylabel("Class", fontsize=12)

    fig.tight_layout()
    if name is not None:
        plt.savefig(f"scenario_{name}.png")
