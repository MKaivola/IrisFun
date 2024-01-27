import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB

# Metrics
from sklearn.metrics import confusion_matrix, accuracy_score

import os

# User methods
import utils


# Read train and test data
vowel_train, vowel_test = pandas.read_csv("vowel/vowel_train.csv"), \
    pandas.read_csv("vowel/vowel_test.csv")

# Remove redundant row index

vowel_train = vowel_train.drop(["row.names"], axis = 1)
vowel_test = vowel_test.drop(["row.names"], axis = 1)

# Descriptive statistics

# Label distribution

label_counts = vowel_train["y"].value_counts(normalize=True)
label_fig, label_ax = plt.subplots()

label_ax.bar(label_counts.index.tolist(), label_counts)
label_ax.set_ylabel("Frequency")
label_ax.set_xlabel("Label")
label_ax.set_title("Label distribution")

label_fig.savefig("vowel/label_hist.pdf")

# Label distribution is uniform

# Analyze label distributions via PCA

vowel_train_features = vowel_train.drop('y', axis = 1)
vowel_train_labels = vowel_train['y']

vowel_PCA = PCA()
vowel_PCA.fit(vowel_train_features)

# Plot the cumulative explained variance as a barplot

explained_vars_cum = np.array(vowel_PCA.explained_variance_ratio_).cumsum()

var_bar_fig, var_bar_ax = plt.subplots()

var_bar_ax.bar(np.arange(1, vowel_PCA.n_components_ + 1 ), explained_vars_cum)
var_bar_ax.set_xlabel("Principal component")
var_bar_ax.set_ylabel("Cumulative prop. of variance accounted")
var_bar_ax.set_title("Distribution of variance over PCs")

var_bar_ax.axhline(0.9, ls = '--', c = 'red')

var_bar_fig.savefig("vowel/PCA_var_dist.pdf")

# Plot 2D rep
vowel_feat_PCA_projec = vowel_PCA.transform(vowel_train_features)

pca_fig, pca_ax = plt.subplots()

pca_ax.scatter(vowel_feat_PCA_projec[:,0], vowel_feat_PCA_projec[:,1], c = vowel_train_labels)
pca_ax.set_ylabel("PC 2")
pca_ax.set_xlabel("PC 1")
pca_ax.set_title(f'Feature projection via PCA, Prop. of total var: {explained_vars_cum[1].round(2)} ')

pca_fig.savefig("vowel/PC_2D.pdf")

# Naive Bayes classifier

# Extract test features and labels

vowel_test_features = vowel_test.drop('y', axis = 1)
vowel_test_labels = vowel_test['y']

naive_bayes = GaussianNB() # Label distribution is uniform
naive_bayes.fit(vowel_train_features, vowel_train_labels)

bayes_pred_labels = naive_bayes.predict(vowel_test_features)

bayes_acc = accuracy_score(vowel_test_labels, bayes_pred_labels)

# Plot the confusion matrix of the Naive Bayes classifier

conf_matr = confusion_matrix(vowel_test_labels, bayes_pred_labels, normalize = "true")

vowel_labels = np.arange(1, np.max(vowel_test_labels) + 1)
utils.heat_map_plot(conf_matr, vowel_labels, vowel_labels, "vowel/conf_matr_naive_bayes.pdf")
