import pytest
from sklearn.dummy import DummyClassifier
import numpy as np
import pandas as pd
import skops.io as sio

import utils

class UntrustedType():
    def malicious_code(self):
        pass

@pytest.fixture
def LearnedClassifier():
    return DummyClassifier()

@pytest.fixture
def UntrustedClass():
    return UntrustedType()

class TestSaveModel():

    def test_no_model_name_specified(self, LearnedClassifier, tmp_path):

        with pytest.raises(ValueError):
            utils.save_model(tmp_path.as_posix(), LearnedClassifier)

    def test_new_folder_created(self, LearnedClassifier, tmp_path):

        full_path = tmp_path / "model_folder" 

        utils.save_model((full_path / "test_model.skops").as_posix(), LearnedClassifier)

        assert full_path.exists()

class TestLoadModel():
    def test_incorrect_model_format(self, tmp_path, LearnedClassifier):

        full_path = tmp_path / "vovel_model.dat"

        sio.dump(LearnedClassifier, full_path)

        with pytest.raises(ValueError):
            utils.load_model(full_path.as_posix())

    def test_untrusted_model_interrupt(self, tmp_path, UntrustedClass, monkeypatch):
        
        full_path = tmp_path / "untrusted_model.skops"

        sio.dump(UntrustedClass, full_path) 

        with pytest.raises(KeyboardInterrupt):
            monkeypatch.setattr("builtins.input", lambda _: 'n')
            utils.load_model(full_path.as_posix())

@pytest.fixture
def panda_dataframe():

    n = 10

    df = pd.DataFrame(np.zeros((n,n)))

    df = df.rename(columns = {0:'row.names', 1:'y'})

    df['y'] = np.random.randint(low = 0, high = 5, size = n)

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

    def test_different_amounts_of_features(self, panda_dataframe):

        df = panda_dataframe.assign(new_feature = np.repeat(0.0, panda_dataframe.shape[0]))

        with pytest.raises(ValueError):
            utils.validate_input(df, panda_dataframe)

    def test_unobserved_class(self, panda_dataframe):

        df = panda_dataframe.copy()

        df['y'] = np.random.randint(low = 6, high = 10, size = panda_dataframe.shape[0])

        with pytest.raises(ValueError):
            utils.validate_input(df, panda_dataframe)


    