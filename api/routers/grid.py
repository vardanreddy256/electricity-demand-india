"""Grid planning router — capacity gap, renewables, scenarios, load curve."""
import numpy as np
import pandas as pd
from fastapi import APIRouter, Query
from api.dependencies import get_cleaned
from src.grid_planning import (
    get_capacity_summary, capacity_gap_analysis, renewable_analysis,
    scenario_projections, load_duration_curve, peak_heatmap,
)

router = APIRouter()


@router.get("/capacity-summary")
def grid_capacity_summary():
    return get_capacity_summary()


@router.get("/capacity-gap")
def grid_capacity_gap():
    df = get_cleaned()
    return capacity_gap_analysis(df)


@router.get("/renewables")
def grid_renewables():
    df = get_cleaned()
    return renewable_analysis(df)


@router.get("/scenarios")
def grid_scenarios(growth_low: float = Query(0.04), growth_base: float = Query(0.065), growth_high: float = Query(0.09)):
    df = get_cleaned()
    base_demand = float(df["National Hourly Demand"].tail(8760).mean())
    from src.grid_planning import INDIA_GRID
    scenarios = [
        {"name": "Conservative", "rate": growth_low, "color": "#10b981"},
        {"name": "Base Case", "rate": growth_base, "color": "#6366f1"},
        {"name": "High Growth", "rate": growth_high, "color": "#f59e0b"},
    ]
    years = list(range(2024, 2036))
    result = []
    for s in scenarios:
        demand = [round(base_demand * ((1 + s["rate"]) ** (yr - 2024))) for yr in years]
        result.append({"name": s["name"], "rate": s["rate"], "color": s["color"], "years": years, "demand": demand})
    return {
        "base_demand_MW": round(base_demand, 1),
        "scenarios": result,
        "renewable_targets": {
            "solar_2030_MW": INDIA_GRID["target_renewable_2030_MW"],
            "total_2030_MW": INDIA_GRID["target_total_2030_MW"],
        },
    }


@router.get("/load-curve")
def grid_load_curve():
    df = get_cleaned()
    return load_duration_curve(df["National Hourly Demand"])


@router.get("/peak-heatmap")
def grid_peak_heatmap():
    df = get_cleaned()
    return peak_heatmap(df)


@router.get("/seasonal-analysis")
def grid_seasonal():
    df = get_cleaned()
    g = df.groupby("season")["National Hourly Demand"].agg(["mean", "max", "min"]).reset_index()
    g.columns = ["season", "avg", "peak", "min"]
    return {
        "seasons": g["season"].tolist(),
        "avg": g["avg"].round(1).tolist(),
        "peak": g["peak"].round(1).tolist(),
        "min": g["min"].round(1).tolist(),
    }
