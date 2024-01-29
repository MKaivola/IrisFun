import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import RocCurveDisplay

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

def roc_multiclass_plot(true_labels_train: np.ndarray, true_labels_test: np.ndarray, 
                        predict_scores_test: np.ndarray, method: str,
                        filename: str):
    """ 
    Plot a micro-averaged One-vs-Rest (OvR) ROC curve and save the figure to file system
    
    Arguments
    ---------
    true_labels_train
        A 2D array containing the true labels of the training set samples
    true_labels_test
        A 2D array containing the true labels of the test set samples
    predict_scores_test
        A 2D array containing the predicted scores of the test set samples
    method:
        A string containing the name of the classification method
    filename:
        A string specifying the path to save the figure

    """

    # Binarize the multiclass labels
    binarizer = LabelBinarizer().fit(true_labels_train)
    binarized_labels_test = binarizer.transform(true_labels_test)

    # Compute the micro-averaged OvR ROC curve

    roc_plot = RocCurveDisplay.from_predictions(y_true = binarized_labels_test.ravel(),
                                                y_pred = predict_scores_test.ravel(),
                                                name = "Micro-averaged OvR",
                                                plot_chance_level = True,
                                                color = 'blue')
    roc_plot.ax_.set(
        xlabel = "False positive rate",
        ylabel = "True positive rate",
        title = f"Method: {method}"
    )

    roc_plot.figure_.savefig(filename)