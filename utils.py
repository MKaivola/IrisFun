import matplotlib.pyplot as plt
import numpy as np

def heat_map_plot(data_array: np.ndarray, xlabels: list, ylabels: list, filename: str):
    """
    Construct a heat map and save to file system

    Arguments
    ---------
    data_array
        2D array of numbers
    xlabels
        A list specifying the xtick labels
    ylabels
        A list specifying the ytick labels
    filename
        A string specifying the path to save the figure
    """

    fig, ax = plt.subplots()

    ax.imshow(data_array)

    ax.set_xticks(ticks = np.arange(len(xlabels)), labels = xlabels)
    ax.set_yticks(ticks = np.arange(len(ylabels)), labels = ylabels)

    ax.tick_params(axis = 'x', top = True, labeltop = True, bottom = False, labelbottom = False)

    data_array_round = np.round(data_array, 2)

    for x in np.arange(len(xlabels)):
        for y in np.arange(len(ylabels)):
            ax.text(x, y, data_array_round[y,x], ha = "center", va = "center",
                    c = "white")

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.xaxis.set_label_position("top")

    ax.set_title("Confusion matrix for Naive Bayes")

    fig.tight_layout()
    fig.savefig(filename)
