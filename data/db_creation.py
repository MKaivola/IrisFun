from sqlalchemy import create_engine, text
from sqlalchemy import MetaData, Table, Column
from sqlalchemy import Integer, Float
from sqlalchemy import insert
import pandas as pd

import data_utils

engine = create_engine("sqlite:///vowel_data.db", echo = False)

### Table Creation (Textual SQL) ###
with engine.begin() as conn:

    conn.execute(text("DROP TABLE IF EXISTS vowel_train"))

    feature_names = ", ".join(["".join(("x_", str(i), " Float")) for i in range(1,11)])

    stmt = "CREATE TABLE IF NOT EXISTS vowel_train (row_name Integer PRIMARY KEY, " \
            "y Integer, " + feature_names + ")"
    result = conn.execute(text(stmt))

### Data Insertion (Textual SQL) ###
train_data_records = data_utils.df_to_list_of_records(csv_filename='vowel_train.csv')

with engine.begin() as conn:
   
    conn.execute(text("INSERT INTO vowel_train (row_name, y, x_1, x_2, x_3, x_4, "
                      "x_5, x_6, x_7, x_8, x_9, x_10) VALUES "
                      "(:row_name, :y, :x_1, :x_2, :x_3, :x_4, "
                      ":x_5, :x_6, :x_7, :x_8, :x_9, :x_10)"),
                      train_data_records)
    
### Database Metadata definition ###
metadata_obj = MetaData()

test_data_table = Table(
    "vowel_test",
    metadata_obj,
    Column("row_name", Integer, primary_key=True),
    Column("y", Integer),
    Column("x_1", Float),
    Column("x_2", Float),
    Column("x_3", Float),
    Column("x_4", Float),
    Column("x_5", Float),
    Column("x_6", Float),
    Column("x_7", Float),
    Column("x_8", Float),
    Column("x_9", Float),
    Column("x_10", Float),
)

# Create remaining tables
metadata_obj.drop_all(engine)
metadata_obj.create_all(engine)

### Data Insertion using Core constructs ###

test_data_records = data_utils.df_to_list_of_records(csv_filename='vowel_test.csv')

test_data_insert = insert(test_data_table)

with engine.begin() as conn:
    conn.execute(test_data_insert,
                 test_data_records)

