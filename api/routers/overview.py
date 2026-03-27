"""Overview router — KPIs, national trend, heatmap."""
import numpy as np
import pandas as pd
from fastapi import APIRouter, Query

from api.dependencies import get_cleaned

router = APIRouter()


@router.get("/kpis")
def get_kpis():
    df = get_cleaned()
    nd = df["National Hourly Demand"]
    # YoY growth (last full year vs previous)
    y23 = df[df["datetime"].dt.year == 2023]["National Hourly Demand"].mean()
    y22 = df[df["datetime"].dt.year == 2022]["National Hourly Demand"].mean()
    yoy = round((y23 - y22) / y22 * 100, 2)
    # Renewable share
    ren = df["solar_gen_MW"] + df["wind_gen_MW"]
    ren_share = round(ren.mean() / nd.mean() * 100, 1)
    # Load factor (avg/peak)
    load_factor = round(nd.mean() / nd.max() * 100, 1)
    # Latest demand (last record)
    latest = float(df.iloc[-1]["National Hourly Demand"])
    return {
        "peak_demand_MW": float(nd.max()),
        "avg_demand_MW": round(float(nd.mean()), 1),
        "load_factor_pct": load_factor,
        "renewable_share_pct": ren_share,
        "yoy_growth_pct": yoy,
        "latest_demand_MW": round(latest, 1),
        "total_records": len(df),
        "date_range": {
            "start": df["datetime"].min().strftime("%Y-%m-%d"),
            "end": df["datetime"].max().strftime("%Y-%m-%d"),
        },
    }


@router.get("/trend")
def get_trend(
    freq: str = Query("D", description="Resample frequency: H/D/W/ME"),
    start: str = Query(None),
    end: str = Query(None),
):
    df = get_cleaned().copy()
    if start:
        df = df[df["datetime"] >= start]
    if end:
        df = df[df["datetime"] <= end]
    df = df.set_index("datetime")
    agg = df["National Hourly Demand"].resample(freq).mean().dropna().reset_index()
    # Limit to 2000 points
    if len(agg) > 2000:
        idx = np.linspace(0, len(agg) - 1, 2000, dtype=int)
        agg = agg.iloc[idx]
    return {
        "timestamps": agg["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
        "demand": agg["National Hourly Demand"].round(1).tolist(),
    }


@router.get("/heatmap")
def get_heatmap():
    df = get_cleaned()
    pivot = df.groupby(["hour", "dayofweek"])["National Hourly Demand"].mean().reset_index()
    return {
        "hour": pivot["hour"].tolist(),
        "day": pivot["dayofweek"].tolist(),
        "demand": pivot["National Hourly Demand"].round(0).tolist(),
    }


@router.get("/yearly-stats")
def get_yearly_stats():
    df = get_cleaned()
    g = df.groupby(df["datetime"].dt.year)["National Hourly Demand"]
    stats = g.agg(["mean", "max", "min"]).reset_index()
    stats.columns = ["year", "avg", "peak", "min"]
    return {
        "years": stats["year"].tolist(),
        "avg": stats["avg"].round(1).tolist(),
        "peak": stats["peak"].round(1).tolist(),
        "min": stats["min"].round(1).tolist(),
    }
