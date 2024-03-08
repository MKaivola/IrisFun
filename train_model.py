import argparse
import os

import pandas as pd
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

    train_data = pd.read_csv(args.filename_data)
    utils.validate_input(train_data)

    X_train, y_train = utils.preprocess_input(train_data)

    if args.filename_test is not None:
        test_data = pd.read_csv(args.filename_test)
        utils.validate_input(test_data, train_data)
    else:
        test_data = None

    model_pipe = Pipeline([("scaler", StandardScaler()), ("LDA", LinearDiscriminantAnalysis(solver = 'eigen'))])

    hyper_param_grid = {
        "LDA__shrinkage": np.concatenate(([0], uniform.rvs(size = 15, random_state = 0), [1]))
    }

    model_cv = GridSearchCV(model_pipe, hyper_param_grid)

    if test_data is not None:
        # Estimate generalization error via test data

        X_test, y_test = utils.preprocess_input(test_data)

        model_cv.fit(X_train, y_train)

        print("Estimating the generalization accuracy score using test data...")
        test_accuracy = model_cv.score(X_test, y_test)

        print(f"Test set accuracy score: {np.round(test_accuracy, 2)}")
    else:
        # Estimate generalization error via nested cross-validation
        
        print("Estimating the generalization accuracy score using cross-validation...")
        model_performance = cross_val_score(model_cv, X_train, y_train)

        print(f"Cross-validated accuracy score: {np.round(model_performance.mean(), 2)}")

    # Learn model using all data
        
    X_all, y_all = utils.preprocess_input(train_data, test_data)

    print("Learning the model using all data...")
    model_cv.fit(X_all, y_all)

    # Save final model
    utils.save_model(args.model_file, model_cv)
