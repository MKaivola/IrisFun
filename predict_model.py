import argparse

import pandas as pd
from sqlalchemy import select, update, bindparam
from sqlalchemy import Null

import utils_predict
from data.db_metadata import VowelDataBase

parser = argparse.ArgumentParser()

parser.add_argument("model_file",
        type=str,
        help="Path to the file where the learned model is stored")

if __name__ == '__main__':
    args = parser.parse_args()

    data_base = VowelDataBase("sqlite:///data/vowel_data.db")

    unlabeled_data_select_args = [data_base.new_data_table.c['row_name',
                                                              'x_1',
                                                              'x_2',
                                                              'x_3',
                                                              'x_4',
                                                              'x_5',
                                                              'x_6',
                                                              'x_7',
                                                              'x_8',
                                                              'x_9',
                                                              'x_10']]
    
    unlabeled_data_where_args = [data_base.new_data_table.c.y_pred == Null()]
    
    list_of_new_rows = data_base.execute_select(unlabeled_data_select_args, 
                                                unlabeled_data_where_args)

    if not list_of_new_rows:
        print('No new data to predict for')
    else:
        unlabeled_data = pd.DataFrame(list_of_new_rows)

        learned_model = utils_predict.load_model(args.model_file)

        utils_predict.validate_input(unlabeled_data, learned_model)

        X = utils_predict.preprocess_input(unlabeled_data)

        predicted_labels = learned_model.predict(X)

        labeled_data = (unlabeled_data.assign(y_pred = predicted_labels)[['row_name', 'y_pred']]
                        .rename(columns={'row_name': 'b_row_name', 'y_pred': 'b_y_pred'})
                        .to_dict(orient='records')
                        )

        label_data_update_args = [data_base.new_data_table]
        label_data_values_args = {'y_pred': bindparam('b_y_pred')}
        label_data_where_args = [data_base.new_data_table.c.row_name == bindparam('b_row_name')]
        
        data_base.execute_update(label_data_update_args,
                                 label_data_values_args,
                                 label_data_where_args,
                                 labeled_data)