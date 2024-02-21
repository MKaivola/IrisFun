import argparse
import os

import pandas
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
import skops.io as sio
from scipy.stats import uniform

import utils

parser = argparse.ArgumentParser()

parser.add_argument("filename_data",
        type=str, 
        help="Path to a csv file containing the training data")
parser.add_argument("model_file",
        type=str,
        help="Path to the file where the learned model is stored")
parser.add_argument("--filename_test",
        type=str, 
        help="Path to a csv file containing the test data")

if __name__ == '__main__':
    args = parser.parse_args()

    train_data = pandas.read_csv(args.filename_data)

    # TODO: Input validation

    X = train_data.drop(['row.names', 'y'], axis = 1)
    y = train_data['y']

    model_pipe = Pipeline([("scaler", StandardScaler()), ("LDA", LinearDiscriminantAnalysis(solver = 'eigen'))])

    hyper_param_grid = {
        "LDA__shrinkage": np.concatenate(([0], uniform.rvs(size = 15, random_state = 0), [1]))
    }

    # Estimate generalization error via nested cross-validation
    model_cv = GridSearchCV(model_pipe, hyper_param_grid)

    print("Estimating the generalization accuracy score using cross-validation...")
    model_performance = cross_val_score(model_cv, X, y)
    print(f"Cross-validated accuracy score: {np.round(model_performance.mean(), 2)}")

    # Learn model using all data
    print("Learning the model using all data...")
    model_cv.fit(X, y)

    # Save final model
    utils.save_model(args.model_file, model_cv)
