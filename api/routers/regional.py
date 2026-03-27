"""Regional demand router."""
import numpy as np
import pandas as pd
from fastapi import APIRouter, Query
from api.dependencies import get_cleaned, REGION_COLS

router = APIRouter()

REGION_COLORS = {
    "Northern": "#6366f1",
    "Western": "#10b981",
    "Eastern": "#f59e0b",
    "Southern": "#ef4444",
    "NorthEastern": "#8b5cf6",
}


@router.get("/breakdown")
def get_breakdown():
    df = get_cleaned()
    region_cols = [col for col in REGION_COLS.values() if col in df.columns]
    latest = df.tail(24)[region_cols].mean()
    total = latest.sum()
    regions, demand, share, colors = [], [], [], []
    for name, col in REGION_COLS.items():
        if col in df.columns:
            val = round(float(latest[col]), 1)
            regions.append(name)
            demand.append(val)
            share.append(round(val / total * 100, 2) if total else 0)
            colors.append(REGION_COLORS.get(name, "#6366f1"))
    return {"regions": regions, "demand": demand, "share": share, "colors": colors}


@router.get("/trend")
def get_region_trend(
    region: str = Query("Northern"),
    freq: str = Query("W"),
    start: str = Query(None),
    end: str = Query(None),
):
    df = get_cleaned().copy()
    col = REGION_COLS.get(region)
    if col is None or col not in df.columns:
        return {"error": f"Unknown region: {region}"}
    if start:
        df = df[df["datetime"] >= start]
    if end:
        df = df[df["datetime"] <= end]
    agg = df.set_index("datetime")[col].resample(freq).mean().dropna().reset_index()
    if len(agg) > 1000:
        idx = np.linspace(0, len(agg) - 1, 1000, dtype=int)
        agg = agg.iloc[idx]
    return {
        "region": region,
        "timestamps": agg["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
        "demand": agg[col].round(1).tolist(),
    }


@router.get("/compare")
def get_region_compare(freq: str = Query("ME")):
    df = get_cleaned().copy()
    df = df.set_index("datetime")
    result = {"dates": []}
    for name, col in REGION_COLS.items():
        if col in df.columns:
            agg = df[col].resample(freq).mean().dropna()
            if not result["dates"]:
                result["dates"] = agg.index.strftime("%Y-%m").tolist()
            result[name] = agg.round(1).tolist()
    return result


@router.get("/stats")
def get_region_stats():
    df = get_cleaned()
    rows = []
    for name, col in REGION_COLS.items():
        if col in df.columns:
            s = df[col]
            rows.append({
                "region": name,
                "avg_MW": round(float(s.mean()), 1),
                "peak_MW": round(float(s.max()), 1),
                "min_MW": round(float(s.min()), 1),
                "share_pct": round(float(s.mean()) / float(df["National Hourly Demand"].mean()) * 100, 1),
            })
    return {"regions": rows}
