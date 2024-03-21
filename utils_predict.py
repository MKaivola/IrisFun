import os

import pandas as pd
import skops.io as sio

import custom_types
from utils import df_col_type_check

def load_model(filepath: str) -> custom_types.Scikit_Classifier:
    """
    Load learned classifier from a given file. 
    Checks for unknown types before loading.

    Arguments
    ---------
    filepath 
        A string specifying the filepath to the model

    """

    if not filepath.endswith('.skops'):
        raise ValueError("File name was not specified correctly: use .skops file extension")

    if not os.path.exists(filepath):
        raise ValueError("The file does not exist")
    
    unknown_types = sio.get_untrusted_types(file = filepath)

    if unknown_types:
        print(f"The following unknown types were detected: {unknown_types}")

        answer = None

        while answer not in ('Y','n'):
            answer = input("Are these types trusted? ([Y]es/[n]o)")

        if answer == 'n':
            raise KeyboardInterrupt("The model file was not loaded, program interrupted")
        
    sio.load(filepath, trusted = unknown_types)

def validate_input(data_matrix: pd.DataFrame, model: custom_types.Scikit_Classifier) -> None:
    """
    Check that the input data frame conforms to the preprocessing assumptions
    Additionally, check that the number of features are the same between the data and the model

    Arguments
    ---------
    input_data
        A pandas dataframe containing new unlabeled data
    model
        A learned sklearn classifier
    """

    col_names = data_matrix.columns

    if 'row.names' not in col_names:
        raise KeyError("'row.names' column is missing from the dataframe")
    
    df_col_type_check(data_matrix.drop('row.names', axis = 1))

    model_features = model.n_features_in_
    data_features = data_matrix.shape[-1] - 1

    if data_features != model_features:
        raise ValueError(f"The number of features in the model and the data are not equal. Expected {model_features}, got {data_features} instead.")
    
def preprocess_input(data_matrix: pd.DataFrame) -> pd.DataFrame:

    X = data_matrix.drop(['row.names'], axis = 1)

    return X