from sqlalchemy import create_engine
from sqlalchemy import text

import pandas as pd

engine = create_engine("sqlite:///vowel_data.db", echo = False)

### Table Creation with textual SQL ###
with engine.begin() as conn:

    conn.execute(text("DROP TABLE IF EXISTS vowel_train"))

    feature_names = ", ".join(["".join(("x_", str(i), " REAL")) for i in range(1,11)])

    stmt = "CREATE TABLE IF NOT EXISTS vowel_train (row_name INTEGER PRIMARY KEY, " \
            "y INT, " + feature_names + ") WITHOUT ROWID"
    result = conn.execute(text(stmt))
    
train_data = pd.read_csv("vowel_train.csv")

column_name_dict = {text:text.replace(".","_") for text in train_data.columns}
column_name_dict["row.names"] = "row_name"

train_data_dict = train_data.rename(columns = column_name_dict).to_dict(orient = 'records')

### Table Insertion with textual SQL ###
with engine.begin() as conn:
   
    conn.execute(text("INSERT INTO vowel_train (row_name, y, x_1, x_2, x_3, x_4, "
                      "x_5, x_6, x_7, x_8, x_9, x_10) VALUES "
                      "(:row_name, :y, :x_1, :x_2, :x_3, :x_4, "
                      ":x_5, :x_6, :x_7, :x_8, :x_9, :x_10)"),
                      train_data_dict)



    

