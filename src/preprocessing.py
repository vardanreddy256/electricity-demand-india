"""Data preprocessing module for India electricity demand dataset."""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

COLUMN_MAP = {
    "Northen Region Hourly Demand": "Northern Region Hourly Demand",
}

REGION_COLS = [
    "Northern Region Hourly Demand",
    "Western Region Hourly Demand",
    "Eastern Region Hourly Demand",
    "Southern Region Hourly Demand",
    "North-Eastern Region Hourly Demand",
]


def load_raw_data(filepath=None) -> pd.DataFrame:
    if filepath is None:
        csv_files = list(RAW_DIR.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV found in {RAW_DIR}")
        filepath = csv_files[0]
    logger.info(f"Loading: {filepath}")
    df = pd.read_csv(filepath)
    df = df.rename(columns=COLUMN_MAP)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def validate_data(df: pd.DataFrame) -> None:
    assert df.isnull().sum().sum() == 0, "Null values found"
    assert df.duplicated(subset=["datetime"]).sum() == 0, "Duplicate datetimes"
    logger.info(f"Validation passed — {len(df)} rows, {df.shape[1]} cols")


def save_cleaned(df: pd.DataFrame) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "cleaned.parquet"
    df.to_parquet(out, index=False)
    logger.info(f"Saved cleaned → {out}")
    return out


def run() -> pd.DataFrame:
    df = load_raw_data()
    validate_data(df)
    save_cleaned(df)
    return df


if __name__ == "__main__":
    run()
