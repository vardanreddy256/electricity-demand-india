"""Random Forest demand forecasting model."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from .base import BaseForecaster


class RandomForestForecaster(BaseForecaster):
    def __init__(self, params=None):
        super().__init__("RandomForest")
        self.params = params or {
            "n_estimators": 200,
            "max_depth": 20,
            "min_samples_split": 5,
            "random_state": 42,
            "n_jobs": -1,
        }
        self.model = RandomForestRegressor(**self.params)
        self._feature_names = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self._feature_names = list(X_train.columns) if hasattr(X_train, "columns") else None
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def feature_importance(self, feature_names=None) -> dict:
        names = feature_names or self._feature_names or [f"f{i}" for i in range(len(self.model.feature_importances_))]
        imp = self.model.feature_importances_
        pairs = sorted(zip(names, imp.tolist()), key=lambda x: x[1], reverse=True)
        return {k: round(v, 6) for k, v in pairs[:20]}
