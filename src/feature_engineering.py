"""Feature engineering for electricity demand forecasting."""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

TARGET = "National Hourly Demand"
LAG_HOURS = [1, 24, 48, 168]

FEATURE_COLS = [
    "lag_1h", "lag_24h", "lag_48h", "lag_168h",
    "rolling_mean_24h", "rolling_std_24h", "rolling_mean_168h",
    "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos",
    "temperature_C", "humidity_percent", "heat_index_C",
    "solar_gen_MW", "wind_gen_MW", "weekend_flag", "is_holiday",
    "total_renewable_MW", "renewable_penetration",
    "season_Monsoon", "season_Post-Monsoon", "season_Summer", "season_Winter",
]


def add_cyclical(df):
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_lags(df):
    df = df.copy()
    for lag in LAG_HOURS:
        df[f"lag_{lag}h"] = df[TARGET].shift(lag)
    return df


def add_rolling(df):
    df = df.copy()
    s = df[TARGET].shift(1)
    df["rolling_mean_24h"] = s.rolling(24).mean()
    df["rolling_std_24h"] = s.rolling(24).std()
    df["rolling_mean_168h"] = s.rolling(168).mean()
    return df


def add_holidays(df):
    df = df.copy()
    try:
        import holidays as hlib
        india = hlib.India()
        df["is_holiday"] = df["datetime"].dt.date.apply(lambda d: int(d in india))
    except Exception:
        major = set()
        for yr in range(2019, 2026):
            for m, d in [(1,26),(8,15),(10,2),(11,14),(12,25)]:
                major.add(f"{yr}-{m:02d}-{d:02d}")
        df["is_holiday"] = df["datetime"].dt.strftime("%Y-%m-%d").isin(major).astype(int)
    return df


def add_renewable(df):
    df = df.copy()
    df["total_renewable_MW"] = df["solar_gen_MW"] + df["wind_gen_MW"]
    df["renewable_penetration"] = df["total_renewable_MW"] / df[TARGET]
    return df


def add_season_dummies(df):
    df = df.copy()
    dummies = pd.get_dummies(df["season"], prefix="season")
    for col in ["season_Monsoon", "season_Post-Monsoon", "season_Summer", "season_Winter"]:
        if col not in dummies.columns:
            dummies[col] = 0
    return pd.concat([df, dummies], axis=1)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_cyclical(df)
    df = add_lags(df)
    df = add_rolling(df)
    df = add_holidays(df)
    df = add_renewable(df)
    df = add_season_dummies(df)
    df = df.dropna().reset_index(drop=True)
    return df


def get_feature_columns():
    return FEATURE_COLS


def run() -> pd.DataFrame:
    cleaned = PROCESSED_DIR / "cleaned.parquet"
    df = pd.read_parquet(cleaned)
    df = build_features(df)
    out = PROCESSED_DIR / "features.parquet"
    df.to_parquet(out, index=False)
    logger.info(f"Features saved → {out}  shape={df.shape}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
