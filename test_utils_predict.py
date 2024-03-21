import pytest
from sklearn.dummy import DummyClassifier
import pandas as pd
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