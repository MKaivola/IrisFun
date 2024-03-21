import os
import pandas as pd

import pandas as pd
import numpy as np
import skops.io as sio

import custom_types

def save_model(filepath: str, model: custom_types.Scikit_Classifier) -> None:
    """
    Save learned classifier to a given filepath

    Arguments
    ---------
    filepath 
        A string specifying the filepath
    model
        A scikit learn classifier to be saved

    """

    if not filepath.endswith('.skops'):
        raise ValueError("File name was not specified correctly: use .skops file extension")
    
    dir, _ = os.path.split(filepath)

    if not dir == '' and not os.path.exists(dir):
        os.mkdir(dir)

    sio.dump(model, filepath)

def validate_input(data_matrix: pd.DataFrame, train_data_matrix: pd.DataFrame = None) -> None:
    """
    Check that the input data frame conforms to the preprocessing assumptions
    Optionally, check that the data matrix contains only classes observed during training
    and contains the correct amount of features relative to the training data

    Arguments
    ---------
    data_matrix
        A pandas dataframe to be preprocessed and possibly checked against training data 
    train_data_matrix
        An optional validated pandas dataframe containing the training data

    """
    
    col_names = data_matrix.columns

    if 'row.names' not in col_names or 'y' not in col_names:
        raise KeyError("'row.names' or 'y' column is missing from the dataframe")
    
    dtypes_data = data_matrix.dtypes

    dtypes_is_numeric = dtypes_data.apply(pd.api.types.is_numeric_dtype)

    if not dtypes_is_numeric.drop('row.names').all():
        non_numeric_dims = np.nonzero(~dtypes_is_numeric)
        raise TypeError(f'Columns {col_names[non_numeric_dims]} are not numeric')
    
    if train_data_matrix is not None:

        train_dims = train_data_matrix.shape
        data_dims = data_matrix.shape

        if train_dims[-1] != data_dims[-1]:
            raise ValueError(f"The data matrices have different amounts of features. Expected {train_dims[-1]}, got {data_dims[-1]} instead.")

        data_classes = set(data_matrix['y'])
        train_data_classes = set(train_data_matrix['y'])

        if not data_classes <= train_data_classes:
            new_classes = data_classes.difference(train_data_classes)
            raise ValueError(f"Unobserved classes in test data: {new_classes}")


def preprocess_input(data_matrix_train: pd.DataFrame, data_matrix_test: pd.DataFrame = None) -> tuple[pd.DataFrame, pd.Series]:

    data_matrix = data_matrix_train

    if data_matrix_test is not None:
        data_matrix = pd.concat([data_matrix, data_matrix_test], axis = 0)

    X = data_matrix.drop(['row.names', 'y'], axis = 1)
    y = data_matrix['y']

    return X, y