import pandas as pd


def df_to_list_of_records(csv_filename:str) -> list[dict]:
    """
    Transform a dataframe read from a csv file into a list of dictionaries of records
    
    Arguments
    ---------
    csv_filename
        A string specifying the csv file
    
    """
    train_data = pd.read_csv(csv_filename)

    column_name_dict = {text:text.replace(".","_") for text in train_data.columns}
    column_name_dict["row.names"] = "row_name"

    train_data_dict = train_data.rename(columns = column_name_dict).to_dict(orient = 'records')

    return train_data_dict