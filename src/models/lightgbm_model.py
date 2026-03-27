"""LightGBM demand forecasting model."""
import numpy as np
from .base import BaseForecaster


class LightGBMForecaster(BaseForecaster):
    def __init__(self, params=None):
        super().__init__("LightGBM")
        self.params = params or {
            "n_estimators": 500,
            "max_depth": 7,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        self._feature_names = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        from lightgbm import LGBMRegressor
        self._feature_names = list(X_train.columns) if hasattr(X_train, "columns") else None
        self.model = LGBMRegressor(**self.params)
        callbacks = []
        eval_set = None
        if X_val is not None:
            eval_set = [(X_val, y_val)]
        self.model.fit(X_train, y_train, eval_set=eval_set)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def feature_importance(self, feature_names=None) -> dict:
        names = feature_names or self._feature_names or [f"f{i}" for i in range(len(self.model.feature_importances_))]
        imp = self.model.feature_importances_
        pairs = sorted(zip(names, imp.tolist()), key=lambda x: x[1], reverse=True)
        return {k: round(float(v), 6) for k, v in pairs[:20]}
