"""Abstract base class for all forecasting models."""
from abc import ABC, abstractmethod
import numpy as np
import joblib


class BaseForecaster(ABC):
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.metrics = {}

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        pass

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        pass

    def save(self, path):
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path):
        return joblib.load(path)

    def feature_importance(self, feature_names=None) -> dict:
        return {}
