import pytest
from sklearn.dummy import DummyClassifier
import numpy as np
import pandas as pd

import utils

@pytest.fixture
def LearnedClassifier():
    return DummyClassifier()

class TestSaveModel():

    def test_no_model_name_specified(self, LearnedClassifier, tmp_path):

        with pytest.raises(ValueError):
            utils.save_model(tmp_path.as_posix(), LearnedClassifier)

    def test_new_folder_created(self, LearnedClassifier, tmp_path):

        full_path = tmp_path / "model_folder" 

        utils.save_model((full_path / "test_model.skops").as_posix(), LearnedClassifier)

        assert full_path.exists()

@pytest.fixture
def panda_dataframe():

    df = pd.DataFrame(np.zeros((10,10)))

    df = df.rename(columns = {0:'row.names', 1:'y'})

    return df

class TestValidateInput():

    def test_missing_row_names(self, panda_dataframe):

        df = panda_dataframe.drop('row.names', axis = 1)

        with pytest.raises(KeyError):
            utils.validate_input(df)

    def test_missing_y(self, panda_dataframe):

        df = panda_dataframe.drop('y', axis = 1)

        with pytest.raises(KeyError):
            utils.validate_input(df)

    def test_non_numeric_column(self, panda_dataframe):

        df = panda_dataframe.assign(text = np.repeat('foo', panda_dataframe.shape[0]))

        with pytest.raises(TypeError):
            utils.validate_input(df)

    