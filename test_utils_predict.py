import pytest
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import numpy as np
import skops.io as sio

import utils_predict

@pytest.fixture
def LearnedClassifier():
    return DummyClassifier()

class UntrustedType():
    def malicious_code(self):
        pass

@pytest.fixture
def UntrustedClass():
    return UntrustedType()

class TestLoadModel():
    def test_incorrect_model_format(self, tmp_path, LearnedClassifier):

        full_path = tmp_path / "vovel_model.dat"

        sio.dump(LearnedClassifier, full_path)

        with pytest.raises(ValueError):
            utils_predict.load_model(full_path.as_posix())

    def test_untrusted_model_interrupt(self, tmp_path, UntrustedClass, monkeypatch):
        
        full_path = tmp_path / "untrusted_model.skops"

        sio.dump(UntrustedClass, full_path) 

        with pytest.raises(KeyboardInterrupt):
            monkeypatch.setattr("builtins.input", lambda _: 'n')
            utils_predict.load_model(full_path.as_posix())

@pytest.fixture
def unlabeled_df():

    n = 10

    df = pd.DataFrame(np.random.rand(n,2))

    df = df.rename(columns = {0:'row_name'})

    return df

@pytest.fixture
def dummy_model(unlabeled_df):
    model = LinearDiscriminantAnalysis()

    labels = np.random.randint(low = 0, high = 3, size = unlabeled_df.shape[0])

    model.fit(unlabeled_df.drop('row_name', axis = 1), labels)

    return model


class TestValidateInput():

    def test_missing_row_names(self, unlabeled_df, dummy_model):

        df = unlabeled_df.drop('row_name', axis = 1)

        with pytest.raises(KeyError):
            utils_predict.validate_input(df, dummy_model)

    def test_non_numeric_column(self, unlabeled_df, dummy_model):

        df = unlabeled_df.assign(text = np.repeat('foo', unlabeled_df.shape[0]))

        with pytest.raises(TypeError):
            utils_predict.validate_input(df, dummy_model)

    def test_different_amounts_of_features(self, unlabeled_df, dummy_model):

        df = unlabeled_df.assign(new_feature = np.repeat(0.0, unlabeled_df.shape[0]))

        with pytest.raises(ValueError):
            utils_predict.validate_input(df, dummy_model)