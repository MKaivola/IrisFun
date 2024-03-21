import pandas as pd
import numpy as np

def df_col_type_check(data_matrix: pd.DataFrame) -> None:
    """
    Check that all columns are numeric

    Arguments
    ---------
    data_matrix
        A pandas dataframe to be checked
    """

    col_names = data_matrix.columns

    dtypes_data = data_matrix.dtypes

    dtypes_is_numeric = dtypes_data.apply(pd.api.types.is_numeric_dtype)

    if not dtypes_is_numeric.all():
        non_numeric_dims = np.nonzero(~dtypes_is_numeric)
        raise TypeError(f'Columns {col_names[non_numeric_dims]} are not numeric')