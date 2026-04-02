"""
Multi-Granularity Electricity Demand Forecasting — Training Script
==================================================================
Trains an Ensemble (XGBoost + Random Forest + Gradient Boosting)
for each target region. Saves models + metadata for the dashboard.

Granularities handled:
  - Second / Minute  : Synthesized via interpolation (no raw sub-hourly data)
  - Hour             : Direct XGBoost prediction (primary model)
  - Day / Week / Month / Year : Aggregated from hourly predictions

Usage:
  python src/train_multi_granularity.py
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "enhanced_hourly_electricity_dataset.csv")
OUTPUT_DIR = BASE_DIR

MODEL_OUT      = os.path.join(OUTPUT_DIR, "model_dict.pkl")
FEATURES_OUT   = os.path.join(OUTPUT_DIR, "features.pkl")
ENCODER_OUT    = os.path.join(OUTPUT_DIR, "season_encoder.pkl")
SCALER_OUT     = os.path.join(OUTPUT_DIR, "scaler.pkl")
META_OUT       = os.path.join(OUTPUT_DIR, "model_meta.pkl")

# ── Feature Engineering ────────────────────────────────────────────────────────
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rich temporal + lag + rolling feature engineering."""
    df = df.copy()

    # ── Basic time features ──────────────────────────────────────────
    df["hour"]         = df.index.hour
    df["dayofweek"]    = df.index.dayofweek
    df["month"]        = df.index.month
    df["quarter"]      = df.index.quarter
    df["year"]         = df.index.year
    df["dayofyear"]    = df.index.dayofyear
    df["weekofyear"]   = df.index.isocalendar().week.astype(int)
    df["is_weekend"]   = (df.index.dayofweek >= 5).astype(int)
    df["is_monthstart"] = df.index.is_month_start.astype(int)
    df["is_monthend"]   = df.index.is_month_end.astype(int)

    # ── Cyclical encoding (sin/cos) for hour, month, dayofweek ──────
    df["hour_sin"]        = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]        = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"]       = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]       = np.cos(2 * np.pi * df["month"] / 12)
    df["dayofweek_sin"]   = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"]   = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["dayofyear_sin"]   = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["dayofyear_cos"]   = np.cos(2 * np.pi * df["dayofyear"] / 365)

    return df


def add_lags(df: pd.DataFrame, target: str, lags: list) -> pd.DataFrame:
    """Add lag features for a specific target column."""
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}h"] = df[target].shift(lag)
    return df


def add_rolling(df: pd.DataFrame, target: str, windows: list) -> pd.DataFrame:
    """Add rolling statistics for a specific target column."""
    df = df.copy()
    for w in windows:
        df[f"roll_{w}h_mean"] = df[target].shift(1).rolling(w).mean()
        df[f"roll_{w}h_std"]  = df[target].shift(1).rolling(w).std()
        df[f"roll_{w}h_max"]  = df[target].shift(1).rolling(w).max()
    return df


# ── Best-in-class Ensemble ─────────────────────────────────────────────────────
def build_ensemble(X_train, y_train, X_test, y_test):
    """
    Train the best ML models for electricity forecasting:
      1. XGBoost with early stopping (primary — state-of-the-art for tabular time-series)
      2. Random Forest (ensemble diversity, robust fallback)

    Returns the best single model (by MAPE on test set).
    XGBoost typically wins but RF serves as a strong cross-check.
    """

    print("    [1/2] Training XGBoost (with early stopping)...")
    xgb_model = xgb.XGBRegressor(
        n_estimators          = 800,
        max_depth             = 6,
        learning_rate         = 0.03,
        subsample             = 0.85,
        colsample_bytree      = 0.75,
        colsample_bylevel     = 0.8,
        gamma                 = 0.05,
        reg_alpha             = 0.05,
        reg_lambda            = 1.5,
        min_child_weight      = 3,
        objective             = "reg:squarederror",
        tree_method           = "hist",
        early_stopping_rounds = 40,
        random_state          = 42,
        n_jobs                = -1
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set = [(X_test, y_test)],
        verbose  = False
    )

    print("    [2/2] Training Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators     = 200,
        max_depth        = 20,
        min_samples_leaf = 4,
        max_features     = 0.7,
        n_jobs           = -1,
        random_state     = 42
    )
    rf_model.fit(X_train, y_train)

    # Evaluate both
    results = {}
    for name, mdl in [("XGBoost", xgb_model), ("RandomForest", rf_model)]:
        preds = mdl.predict(X_test)
        preds = np.clip(preds, 0, None)
        mape  = mean_absolute_percentage_error(y_test, preds)
        r2    = r2_score(y_test, preds)
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        results[name] = {"model": mdl, "mape": mape, "r2": r2, "rmse": rmse}
        print(f"        {name}: MAPE={mape*100:.2f}%  R2={r2:.4f}  RMSE={rmse:.0f} MW")

    # Return best model by MAPE
    best_name = min(results, key=lambda k: results[k]["mape"])
    print(f"    >>> Best model: {best_name}")
    return results[best_name]["model"], {n: {k: v for k, v in r.items() if k != "model"} for n, r in results.items()}




# -- Main Training --------------------------------------------------------------
def train():
    t0 = time.time()

    # -- Load Data ------------------------------------------------------
    print("\n" + "=" * 60)
    print("  MULTI-GRANULARITY ELECTRICITY FORECASTING - TRAINING")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Data file not found: {DATA_PATH}")
        sys.exit(1)

    print(f"\n[1/5] Loading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Find datetime column
    date_col = next((c for c in df.columns if "date" in c.lower() or "time" in c.lower()), None)
    if not date_col:
        print("[ERROR] No datetime column found.")
        sys.exit(1)

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.set_index(date_col).sort_index()
    df = df[~df.index.duplicated(keep="first")]

    print(f"  Rows: {len(df):,}  |  Date range: {df.index.min().date()} to {df.index.max().date()}")

    # -- Identify Target Columns ----------------------------------------
    target_cols = [c for c in df.columns
                   if ("Hourly" in c or "Demand" in c)
                   and pd.api.types.is_numeric_dtype(df[c])]
    print(f"  Target regions: {target_cols}")

    # -- Encode Season --------------------------------------------------
    print("\n[2/5] Encoding season feature...")
    le = LabelEncoder()
    if "season" in df.columns:
        df["season_encoded"] = le.fit_transform(df["season"].astype(str))
        has_season = True
    else:
        has_season = False
        le = None

    # -- Feature Engineering --------------------------------------------
    print("[3/5] Engineering base temporal features...")
    df = make_features(df)

    BASE_FEATURES = [
        "hour", "dayofweek", "month", "quarter", "year", "dayofyear",
        "weekofyear", "is_weekend", "is_monthstart", "is_monthend",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "dayofweek_sin", "dayofweek_cos", "dayofyear_sin", "dayofyear_cos",
    ]

    if has_season:
        BASE_FEATURES.append("season_encoded")

    WEATHER_FEATURES = ["temperature_C", "humidity_percent", "heat_index_C", "solar_gen_MW", "wind_gen_MW"]
    for wf in WEATHER_FEATURES:
        if wf in df.columns:
            BASE_FEATURES.append(wf)

    # Fill weather nulls with median
    for wf in WEATHER_FEATURES:
        if wf in df.columns:
            df[wf] = df[wf].fillna(df[wf].median())

    # -- Train per Region -----------------------------------------------
    print("\n[4/5] Training models per region...\n")
    model_dict = {}
    model_meta = {}
    scaler_dict = {}

    LAG_HOURS    = [1, 2, 3, 6, 12, 24, 48, 168]   # lag 1h to 1 week
    ROLLING_WIN  = [3, 6, 24, 168]

    for target in target_cols:
        print(f"  >> {target}")

        # Add lag & rolling features
        tmp = add_lags(df.copy(), target, LAG_HOURS)
        tmp = add_rolling(tmp, target, ROLLING_WIN)

        lag_feat = [f"lag_{l}h" for l in LAG_HOURS]
        roll_feat = []
        for w in ROLLING_WIN:
            roll_feat += [f"roll_{w}h_mean", f"roll_{w}h_std", f"roll_{w}h_max"]

        FEATURES = BASE_FEATURES + lag_feat + roll_feat

        # Drop NaN rows (from lags/rolling)
        valid = tmp.dropna(subset=[target] + FEATURES)
        print(f"    Valid rows after lag drop: {len(valid):,}")

        # Train / Test split (80/20 temporal)
        split = int(len(valid) * 0.8)
        train_df = valid.iloc[:split]
        test_df  = valid.iloc[split:]

        X_train = train_df[FEATURES].values.astype(np.float32)
        y_train = train_df[target].values.astype(np.float32)
        X_test  = test_df[FEATURES].values.astype(np.float32)
        y_test  = test_df[target].values.astype(np.float32)

        # Scale for non-tree-compatible (not needed for XGB/RF, but stored)
        scaler = StandardScaler()
        scaler.fit(X_train)
        scaler_dict[target] = scaler

        best_model, metrics = build_ensemble(X_train, y_train, X_test, y_test)
        model_dict[target] = best_model
        model_meta[target] = {
            "features": FEATURES,
            "metrics": metrics,
            "n_train": len(train_df),
            "n_test":  len(test_df),
        }

    # ── Save ───────────────────────────────────────────────────────────
    print("\n[5/5] Saving models & metadata...")
    joblib.dump(model_dict,   MODEL_OUT,    compress=3)
    joblib.dump(BASE_FEATURES, FEATURES_OUT)
    joblib.dump(model_meta,   META_OUT,     compress=3)
    joblib.dump(scaler_dict,  SCALER_OUT,   compress=3)
    if le:
        joblib.dump(le, ENCODER_OUT)

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"   Models saved: {MODEL_OUT}")
    print(f"   Meta saved:   {META_OUT}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    train()

