import os

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

def validate_input(data_matrix: pd.DataFrame) -> None:
    """
    Check that the input data frame conforms to the preprocessing assumptions

    Arguments
    ---------
    data_matrix
        A pandas dataframe to be preprocessed

    """
    
    col_names = data_matrix.columns

    if 'row.names' not in col_names or 'y' not in col_names:
        raise KeyError("'row.names' or 'y' column is missing from the dataframe")
    
    dtypes_data = data_matrix.dtypes

    dtypes_is_numeric = dtypes_data.apply(pd.api.types.is_numeric_dtype)

    if not dtypes_is_numeric.drop('row.names').all():
        non_numeric_dims = np.nonzero(~dtypes_is_numeric)
        raise TypeError(f'Columns {col_names[non_numeric_dims]} are not numeric')
