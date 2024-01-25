import numpy as np
import pandas
import matplotlib.pyplot as plt

import os

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

