"""Forecast router — model predictions vs actuals."""
import numpy as np
import pandas as pd
from fastapi import APIRouter, Query
from api.dependencies import get_cleaned, get_features, get_predictions, get_metrics, load_model, MODELS_DIR
from src.feature_engineering import get_feature_columns
from src.evaluator import calculate_metrics

router = APIRouter()

TARGET = "National Hourly Demand"
MODEL_MAP = {"xgboost": "xgboost", "lightgbm": "lightgbm", "randomforest": "random_forest"}


@router.get("/actuals")
def get_actuals(start: str = Query(None), end: str = Query(None), freq: str = Query("D")):
    df = get_cleaned().copy()
    if start:
        df = df[df["datetime"] >= start]
    if end:
        df = df[df["datetime"] <= end]
    agg = df.set_index("datetime")[TARGET].resample(freq).mean().dropna().reset_index()
    return {
        "timestamps": agg["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
        "demand": agg[TARGET].round(1).tolist(),
    }


@router.get("/predict")
def get_prediction(
    model: str = Query("xgboost", description="xgboost | lightgbm | randomforest"),
    steps: int = Query(168, ge=24, le=720),
):
    df_feat = get_features()
    if df_feat is None:
        return {"error": "Models not trained yet. Run scripts/run_pipeline.py first."}

    model_key = MODEL_MAP.get(model.lower(), "xgboost")
    mdl = load_model(model_key)
    if mdl is None:
        return {"error": f"Model '{model}' not found. Run training first."}

    FEAT = [c for c in get_feature_columns() if c in df_feat.columns]
    # Use test period (2024)
    df_test = df_feat[df_feat["datetime"].dt.year >= 2024].head(steps)
    if len(df_test) == 0:
        df_test = df_feat.tail(steps)

    X = df_test[FEAT]
    y_true = df_test[TARGET].values
    y_pred = mdl.predict(X)

    metrics = calculate_metrics(y_true, y_pred)
    # Simple ±1.5*std confidence band
    err = np.abs(y_true - y_pred)
    ci = float(np.percentile(err, 80))

    return {
        "model": model,
        "steps": len(df_test),
        "timestamps": df_test["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
        "actual": y_true.round(1).tolist(),
        "predicted": y_pred.round(1).tolist(),
        "lower_bound": (y_pred - ci).round(1).tolist(),
        "upper_bound": (y_pred + ci).round(1).tolist(),
        "metrics": metrics,
    }


@router.get("/future")
def get_future_prediction(
    model: str = Query("xgboost", description="xgboost | lightgbm | randomforest"),
    steps: int = Query(168, ge=24, le=720),
):
    df_feat = get_features()
    if df_feat is None:
        return {"error": "Models not trained yet. Run scripts/run_pipeline.py first."}

    model_key = MODEL_MAP.get(model.lower(), "xgboost")
    mdl = load_model(model_key)
    if mdl is None:
        return {"error": f"Model '{model}' not found. Run training first."}

    FEAT = [c for c in get_feature_columns() if c in df_feat.columns]
    
    # To predict true future, we take the last known week of features, 
    # copy it `steps/168` times conceptually, or just shift by 1 week
    # Let's take the absolute last 'steps' rows as naive baseline
    df_last = df_feat.tail(steps).copy()
    
    # Advance the datetime to the future
    last_known_dt = df_feat["datetime"].max()
    future_dates = pd.date_range(start=last_known_dt + pd.Timedelta(hours=1), periods=steps, freq="H")
    
    df_last["datetime"] = future_dates
    X = df_last[FEAT]
    
    y_pred = mdl.predict(X)

    # Estimate pure future confidence band (wider than historical)
    ci = 1500.0  # fallback constant if historical CI isn't mapped
    
    return {
        "model": model,
        "steps": steps,
        "timestamps": df_last["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
        "predicted": y_pred.round(1).tolist(),
        "lower_bound": (y_pred - ci).round(1).tolist(),
        "upper_bound": (y_pred + ci).round(1).tolist(),
    }



@router.get("/feature-importance")
def get_feature_importance(model: str = Query("xgboost")):
    metrics = get_metrics()
    model_key = model
    for k in metrics:
        if k.lower() == model.lower():
            model_key = k
            break
    fi = metrics.get(model_key, {}).get("feature_importance", {})
    if not fi:
        return {"labels": [], "values": []}
    top = list(fi.items())[:15]
    return {
        "labels": [t[0] for t in top],
        "values": [round(t[1], 6) for t in top],
    }
