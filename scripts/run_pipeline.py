"""
End-to-end pipeline: preprocess → feature engineering → train models → save.
Run: python scripts/run_pipeline.py
"""
import sys, json, time, logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import run as do_preprocess
from src.feature_engineering import run as do_features, get_feature_columns
from src.models.xgboost_model import XGBoostForecaster
from src.models.random_forest_model import RandomForestForecaster
from src.evaluator import calculate_metrics

MODELS_DIR = ROOT / "saved_models"
PROCESSED_DIR = ROOT / "data" / "processed"
TARGET = "National Hourly Demand"


def split(df):
    train = df[df["datetime"].dt.year <= 2022]
    val = df[df["datetime"].dt.year == 2023]
    test = df[df["datetime"].dt.year >= 2024]
    return train, val, test


def main():
    logger.info("=" * 60)
    logger.info("ELECTRICITY DEMAND FORECASTING — TRAINING PIPELINE")
    logger.info("=" * 60)

    # Step 1: Preprocess
    logger.info("\n[1/4] Preprocessing...")
    do_preprocess()

    # Step 2: Features
    logger.info("\n[2/4] Engineering features...")
    do_features()

    # Step 3: Load features
    logger.info("\n[3/4] Loading features...")
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"])

    FEAT = [c for c in get_feature_columns() if c in df.columns]
    df_train, df_val, df_test = split(df)

    X_tr, y_tr = df_train[FEAT], df_train[TARGET]
    X_val, y_val = df_val[FEAT], df_val[TARGET]
    X_te, y_te = df_test[FEAT], df_test[TARGET]

    logger.info(f"Train={len(df_train)} | Val={len(df_val)} | Test={len(df_test)}")

    MODELS_DIR.mkdir(exist_ok=True)
    all_metrics = {}
    preds = {
        "datetime": df_test["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
        "actual": y_te.round(2).tolist(),
    }

    # Step 4a: XGBoost
    logger.info("\n[4a] Training XGBoost...")
    t0 = time.time()
    xgb = XGBoostForecaster()
    xgb.fit(X_tr, y_tr, X_val, y_val)
    t = round(time.time() - t0, 1)
    yp = xgb.predict(X_te)
    m = calculate_metrics(y_te, yp)
    m["training_time_s"] = t
    m["feature_importance"] = xgb.feature_importance(FEAT)
    all_metrics["XGBoost"] = m
    preds["XGBoost"] = np.round(yp, 2).tolist()
    xgb.save(MODELS_DIR / "xgboost.pkl")
    logger.info(f"  XGBoost ✓ MAE={m['mae']:.0f} MAPE={m['mape']:.2f}% R²={m['r2']:.4f} [{t}s]")

    # Step 4b: LightGBM (optional)
    try:
        from src.models.lightgbm_model import LightGBMForecaster
        logger.info("\n[4b] Training LightGBM...")
        t0 = time.time()
        lgb = LightGBMForecaster()
        lgb.fit(X_tr, y_tr, X_val, y_val)
        t = round(time.time() - t0, 1)
        yp = lgb.predict(X_te)
        m = calculate_metrics(y_te, yp)
        m["training_time_s"] = t
        m["feature_importance"] = lgb.feature_importance(FEAT)
        all_metrics["LightGBM"] = m
        preds["LightGBM"] = np.round(yp, 2).tolist()
        lgb.save(MODELS_DIR / "lightgbm.pkl")
        logger.info(f"  LightGBM ✓ MAE={m['mae']:.0f} MAPE={m['mape']:.2f}% R²={m['r2']:.4f} [{t}s]")
    except Exception as e:
        logger.warning(f"  LightGBM skipped: {e}")

    # Step 4c: Random Forest
    logger.info("\n[4c] Training Random Forest...")
    t0 = time.time()
    rf = RandomForestForecaster()
    rf.fit(X_tr, y_tr)
    t = round(time.time() - t0, 1)
    yp = rf.predict(X_te)
    m = calculate_metrics(y_te, yp)
    m["training_time_s"] = t
    m["feature_importance"] = rf.feature_importance(FEAT)
    all_metrics["RandomForest"] = m
    preds["RandomForest"] = np.round(yp, 2).tolist()
    rf.save(MODELS_DIR / "random_forest.pkl")
    logger.info(f"  RandomForest ✓ MAE={m['mae']:.0f} MAPE={m['mape']:.2f}% R²={m['r2']:.4f} [{t}s]")

    # Save metrics & predictions
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    pd.DataFrame(preds).to_parquet(MODELS_DIR / "test_predictions.parquet", index=False)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE ✓")
    logger.info(f"Models saved to: {MODELS_DIR}")
    logger.info("=" * 60)

    best = min(all_metrics.items(), key=lambda x: x[1]["mape"])
    logger.info(f"\nBest model: {best[0]} (MAPE={best[1]['mape']:.2f}%)")


if __name__ == "__main__":
    main()
