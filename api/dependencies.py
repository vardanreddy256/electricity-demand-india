"""Shared state: data + loaded models available to all routers."""
import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "saved_models"

# ── Cached data (loaded once at startup) ──────────────────────────────────────
_df_cleaned: pd.DataFrame | None = None
_df_features: pd.DataFrame | None = None
_metrics: dict | None = None
_predictions: pd.DataFrame | None = None


def get_cleaned() -> pd.DataFrame:
    global _df_cleaned
    if _df_cleaned is None:
        path = PROCESSED_DIR / "cleaned.parquet"
        if not path.exists():
            from src.preprocessing import run
            run()
        _df_cleaned = pd.read_parquet(path)
        _df_cleaned["datetime"] = pd.to_datetime(_df_cleaned["datetime"])
        # Fix column typo if present
        if "Northen Region Hourly Demand" in _df_cleaned.columns:
            _df_cleaned = _df_cleaned.rename(columns={"Northen Region Hourly Demand": "Northern Region Hourly Demand"})
    return _df_cleaned


def get_features() -> pd.DataFrame:
    global _df_features
    if _df_features is None:
        path = PROCESSED_DIR / "features.parquet"
        if not path.exists():
            return None
        _df_features = pd.read_parquet(path)
        _df_features["datetime"] = pd.to_datetime(_df_features["datetime"])
    return _df_features


def get_metrics() -> dict:
    global _metrics
    if _metrics is None:
        path = MODELS_DIR / "metrics.json"
        if path.exists():
            with open(path) as f:
                _metrics = json.load(f)
        else:
            _metrics = {}
    return _metrics


def get_predictions() -> pd.DataFrame | None:
    global _predictions
    if _predictions is None:
        path = MODELS_DIR / "test_predictions.parquet"
        if path.exists():
            _predictions = pd.read_parquet(path)
    return _predictions


def load_model(name: str):
    """Load a saved model by name (xgboost / lightgbm / random_forest)."""
    import joblib
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


REGION_COLS = {
    "Northern": "Northern Region Hourly Demand",
    "Western": "Western Region Hourly Demand",
    "Eastern": "Eastern Region Hourly Demand",
    "Southern": "Southern Region Hourly Demand",
    "NorthEastern": "North-Eastern Region Hourly Demand",
}
