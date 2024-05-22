import argparse
import os

import pandas as pd
import numpy as np
from sqlalchemy import select
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
import skops.io as sio
from scipy.stats import uniform

import utils_train
from data.db_metadata import VowelDataBase

parser = argparse.ArgumentParser()

parser.add_argument("model_file",
        type=str,
        help="Path to the file where the learned model will be stored")
parser.add_argument("--use_test_data",
        action="store_true", 
        help="Flag to specify whether to use test data to evaluate model")

if __name__ == '__main__':
    args = parser.parse_args()

    data_base = VowelDataBase("sqlite:///data/vowel_data.db")

    select_train_data = select(data_base.train_data_table.c['y',
                                                          'x_1',
                                                          'x_2',
                                                          'x_3',
                                                          'x_4',
                                                          'x_5',
                                                          'x_6',
                                                          'x_7',
                                                          'x_8',
                                                          'x_9',
                                                          'x_10'])

    list_of_train_rows = data_base.execute_return(select_train_data)

    train_data = pd.DataFrame(list_of_train_rows)
    utils_train.validate_input(train_data)

    X_train, y_train = utils_train.preprocess_input(train_data)

    if args.use_test_data:

        select_test_data = select(data_base.test_data_table.c['y',
                                                          'x_1',
                                                          'x_2',
                                                          'x_3',
                                                          'x_4',
                                                          'x_5',
                                                          'x_6',
                                                          'x_7',
                                                          'x_8',
                                                          'x_9',
                                                          'x_10'])

        list_of_test_rows = data_base.execute_return(select_test_data)

        test_data = pd.DataFrame(list_of_test_rows)
        
        utils_train.validate_input(test_data, train_data)
    else:
        test_data = None

    model_pipe = Pipeline([("scaler", StandardScaler()), ("LDA", LinearDiscriminantAnalysis(solver = 'eigen'))])

    hyper_param_grid = {
        "LDA__shrinkage": np.concatenate(([0], uniform.rvs(size = 15, random_state = 0), [1]))
    }

    model_cv = GridSearchCV(model_pipe, hyper_param_grid)

    if test_data is not None:
        # Estimate generalization error via test data

        X_test, y_test = utils_train.preprocess_input(test_data)

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
        
    X_all, y_all = utils_train.preprocess_input(train_data, test_data)

    print("Learning the model using all data...")
    model_cv.fit(X_all, y_all)

    # Save final model
    utils_train.save_model(args.model_file, model_cv)
