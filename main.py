# Data processing and plotting
import numpy as np
import pandas
import matplotlib.pyplot as plt

# Preprocessing/Feature extraction
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Models
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

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

# Preprocessing 

vowel_test_features = vowel_test.drop('y', axis = 1)
vowel_test_labels = vowel_test['y']

std_scaler = StandardScaler().fit(vowel_train_features)

vowel_train_feat_std = std_scaler.transform(vowel_train_features)
vowel_test_feat_std = std_scaler.transform(vowel_test_features)

# Modeling

model_dict = {}

# Naive Bayes classifier

naive_bayes = GaussianNB() # Label distribution is uniform
model_dict["Naive Bayes"] = naive_bayes
naive_bayes.fit(vowel_train_feat_std, vowel_train_labels)

bayes_pred_labels = naive_bayes.predict(vowel_test_feat_std)

bayes_acc = accuracy_score(vowel_test_labels, bayes_pred_labels)

# Plot the confusion matrix of the Naive Bayes classifier

conf_matr = confusion_matrix(vowel_test_labels, bayes_pred_labels, normalize = "true")

vowel_labels = np.arange(1, np.max(vowel_test_labels) + 1)
utils.heat_map_plot(conf_matr, vowel_labels, vowel_labels, "vowel/conf_matr_naive_bayes.pdf")

# Plot the ROC curve 

naive_bayes_test_probs = naive_bayes.predict_proba(vowel_test_feat_std)

utils.roc_multiclass_plot(vowel_train_labels, vowel_test_labels,
                          naive_bayes_test_probs,
                          "Naive Bayes", "vowel/micro_avg_ROC_naive_bayes.pdf")


# LDA and QDA

lda_model = LinearDiscriminantAnalysis()
model_dict["LDA"] = lda_model
lda_model.fit(vowel_train_feat_std, vowel_train_labels)

# 2D representation of the data using LDA dimensionality reduction

test_data_LDA_proj = lda_model.transform(vowel_test_feat_std)

lda_fig, lda_ax = plt.subplots()

lda_ax.scatter(test_data_LDA_proj[:,0], test_data_LDA_proj[:,1], c = vowel_test_labels)
lda_ax.set_ylabel("Canonical component 2")
lda_ax.set_xlabel("Canonical component 1")

lda_fig.savefig("vowel/LDA_proj.pdf")

# Logistic Regression

log_reg = LogisticRegression(penalty = "l2", max_iter = 200)
log_reg.fit(vowel_train_feat_std, vowel_train_labels)

log_reg_pred_labels = log_reg.predict(vowel_test_feat_std)

log_reg_test_probs = log_reg.predict_proba(vowel_test_feat_std)

utils.roc_multiclass_plot(vowel_train_labels, vowel_test_labels,
                          log_reg_test_probs,
                          "Logistic Regression", "vowel/micro_avg_ROC_log_reg.pdf")


# Print the accuracy of the evaluated models

acc_output = "Model accuracies: "

for model_name, model in model_dict.items():
    model_acc = model.score(vowel_test_feat_std, vowel_test_labels)
    acc_output = acc_output + f"{model_name} {round(model_acc, 2)}, "

print(acc_output)



