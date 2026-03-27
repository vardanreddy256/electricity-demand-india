"""Model comparison router."""
import numpy as np
import pandas as pd
from fastapi import APIRouter
from api.dependencies import get_metrics, get_predictions

router = APIRouter()


@router.get("/metrics")
def compare_metrics():
    metrics = get_metrics()
    if not metrics:
        return {"error": "No trained models found. Run scripts/run_pipeline.py first.", "models": []}

    rows = []
    best_mape = float("inf")
    best_model = None
    for name, m in metrics.items():
        rows.append({
            "name": name,
            "mae": m.get("mae"),
            "rmse": m.get("rmse"),
            "mape": m.get("mape"),
            "r2": m.get("r2"),
            "training_time_s": m.get("training_time_s"),
        })
        if m.get("mape", float("inf")) < best_mape:
            best_mape = m["mape"]
            best_model = name

    return {"models": rows, "best_model": best_model}


@router.get("/predictions")
def compare_predictions(limit: int = 168):
    preds = get_predictions()
    if preds is None:
        return {"error": "No predictions found. Run training first.", "data": {}}
    df = preds.head(limit)
    result = {"timestamps": df["datetime"].tolist(), "actual": df["actual"].round(1).tolist()}
    for col in df.columns:
        if col not in ("datetime", "actual"):
            result[col] = df[col].round(1).tolist()
    return result


@router.get("/radar")
def compare_radar():
    """Normalize metrics for radar chart (higher = better)."""
    metrics = get_metrics()
    if not metrics:
        return {"models": [], "metrics": [], "data": []}

    model_names = list(metrics.keys())
    metric_keys = ["r2", "mape", "mae", "rmse"]
    metric_labels = ["R² Score", "MAPE (inv)", "MAE (inv)", "RMSE (inv)"]

    # Collect raw values
    raw = {k: [metrics[m].get(k, 0) for m in model_names] for k in metric_keys}

    # Normalize 0–1 (for mape/mae/rmse: lower is better → invert)
    def norm_higher_better(vals):
        mn, mx = min(vals), max(vals)
        return [round((v - mn) / (mx - mn + 1e-9), 3) for v in vals]

    def norm_lower_better(vals):
        mn, mx = min(vals), max(vals)
        return [round(1 - (v - mn) / (mx - mn + 1e-9), 3) for v in vals]

    normalized = {
        "r2": norm_higher_better(raw["r2"]),
        "mape": norm_lower_better(raw["mape"]),
        "mae": norm_lower_better(raw["mae"]),
        "rmse": norm_lower_better(raw["rmse"]),
    }

    data = []
    for i, name in enumerate(model_names):
        data.append({
            "model": name,
            "values": [normalized[k][i] for k in metric_keys],
        })

    return {"models": model_names, "metric_labels": metric_labels, "data": data}
