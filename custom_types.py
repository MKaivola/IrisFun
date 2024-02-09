from typing import Protocol

import numpy as np

class Scikit_Dimen_Reduct(Protocol):
    def transform(self, X: np.ndarray) -> np.ndarray: ...

class Scikit_Classifier(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...