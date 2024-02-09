import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import RocCurveDisplay

import custom_types

def heat_map_plot(data_array: np.ndarray, xlabels: list, ylabels: list, filename: str) -> None:
    """
    Construct a heat map and save to file system

    Arguments
    ---------
    data_array
        2D array of numbers
    xlabels
        List specifying the xtick labels
    ylabels
        List specifying the ytick labels
    filename
        String specifying the path to save the figure
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

def roc_multiclass_plot(true_labels_train: np.ndarray, true_labels_test: np.ndarray, features_test: np.ndarray,
                        model_dict: dict,
                        filename: str) -> None:
    """ 
    Plot a micro-averaged One-vs-Rest (OvR) ROC curve and save the figure to file system
    
    Arguments
    ---------
    true_labels_train
        1D array containing the true labels of the training set samples
    true_labels_test
        1D array containing the true labels of the test set samples
    features_test
        2D array containing the features of the test set samples
    model_dict:
        Dictionary containing (model_name:str, model) key value pairs
        which define a model name string and sklearn model pair
    filename:
        String specifying the path to save the figure

    """

    # Binarize the multiclass labels
    binarizer = LabelBinarizer().fit(true_labels_train)
    binarized_labels_test = binarizer.transform(true_labels_test)

    # Compute the micro-averaged OvR ROC curve for each model

    figure, ax = plt.subplots(figsize = (10,10))

    for model_name, model in model_dict.items():

        prob_predictions = model.predict_proba(features_test)

        RocCurveDisplay.from_predictions(y_true = binarized_labels_test.ravel(),
                                                y_pred = prob_predictions.ravel(),
                                                name = f"Micro-Avg OvR: {model_name}",
                                                ax = ax)
   
    ax.set(
        xlabel = "False positive rate",
        ylabel = "True positive rate",
        title = "OvR ROC curves for vowel test data"
    )

    figure.savefig(filename)

def dimen_reduct_plot(feat_mat: np.ndarray, labels: np.ndarray, method: custom_types.Scikit_Dimen_Reduct,
                     xlabel: str, ylabel: str, title: str, filename: str) -> None:
    """"
    Plot a 2D representation of the feature matrix, colored using the true labels
    
    Arguments
    ---------
    feat_mat
        2D array containing the features of the samples
    labels
        1D array containing the true labels of the samples
    method
        scikit-learn class having a transform dimensionality reduction method
    xlabel
        String containing the x axis label of the figure
    yabel
        String containing the y axis label of the figure
    title
        String containing the title of the figure
    filename
        String specifying the path to save the figure

    """

    transform_mat = method.transform(feat_mat)

    fig, ax = plt.subplots()

    ax.scatter(transform_mat[:,0], transform_mat[:,1], c = labels)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    fig.savefig(filename)