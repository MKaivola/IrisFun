import pytest
from sklearn.dummy import DummyClassifier

import utils

@pytest.fixture
def LearnedClassifier():
    return DummyClassifier()

def test_no_model_name_specified(LearnedClassifier, tmp_path):

    with pytest.raises(ValueError):
        utils.save_model(tmp_path.as_posix(), LearnedClassifier)

def test_new_folder_created(LearnedClassifier, tmp_path):

    full_path = tmp_path / "model_folder" 

    utils.save_model((full_path / "test_model.skops").as_posix(), LearnedClassifier)

    assert full_path.exists()

