"""Grid planning analysis constants and functions for India."""
import pandas as pd
import numpy as np

# India installed capacity (approx, MW) — based on CEA data
INDIA_GRID = {
    "total_installed_MW": 480000,
    "thermal_MW": 235000,
    "hydro_MW": 47000,
    "solar_MW": 80000,
    "wind_MW": 44000,
    "other_renewable_MW": 12000,
    "nuclear_MW": 7000,
    "target_renewable_2030_MW": 500000,
    "target_total_2030_MW": 900000,
    "td_loss_pct": 0.19,
}


def get_capacity_summary() -> dict:
    g = INDIA_GRID
    ren = g["solar_MW"] + g["wind_MW"] + g["hydro_MW"] + g["other_renewable_MW"]
    return {
        "total_installed_MW": g["total_installed_MW"],
        "renewable_MW": ren,
        "thermal_MW": g["thermal_MW"],
        "nuclear_MW": g["nuclear_MW"],
        "renewable_pct": round(ren / g["total_installed_MW"] * 100, 1),
        "solar_MW": g["solar_MW"],
        "wind_MW": g["wind_MW"],
        "hydro_MW": g["hydro_MW"],
        "target_renewable_2030_MW": g["target_renewable_2030_MW"],
        "target_total_2030_MW": g["target_total_2030_MW"],
    }


def capacity_gap_analysis(df: pd.DataFrame) -> dict:
    """Daily peak demand vs estimated installed capacity timeline."""
    daily = df.set_index("datetime").resample("D")["National Hourly Demand"].agg(["mean", "max"]).reset_index()
    daily.columns = ["date", "avg_demand", "peak_demand"]
    n = len(daily)
    # India capacity grew from ~360 GW (2019) to ~480 GW (2024)
    daily["installed_cap"] = np.linspace(360000, 480000, n)
    daily["gap_MW"] = (daily["installed_cap"] - daily["peak_demand"]).round(0)
    daily["gap_pct"] = (daily["gap_MW"] / daily["installed_cap"] * 100).round(2)
    # Sample 500 points for chart
    idx = np.linspace(0, n - 1, min(500, n), dtype=int)
    d = daily.iloc[idx]
    return {
        "dates": d["date"].dt.strftime("%Y-%m-%d").tolist(),
        "peak_demand": d["peak_demand"].round(0).tolist(),
        "installed_capacity": d["installed_cap"].round(0).tolist(),
        "gap_MW": d["gap_MW"].tolist(),
        "gap_pct": d["gap_pct"].tolist(),
    }


def renewable_analysis(df: pd.DataFrame) -> dict:
    """Monthly renewable generation trends."""
    df2 = df.copy()
    df2 = df2.set_index("datetime")
    monthly = df2[["solar_gen_MW", "wind_gen_MW", "National Hourly Demand"]].resample("ME").mean().reset_index()
    monthly["total_renewable"] = monthly["solar_gen_MW"] + monthly["wind_gen_MW"]
    monthly["renewable_share"] = (monthly["total_renewable"] / monthly["National Hourly Demand"] * 100).round(2)
    return {
        "dates": monthly["datetime"].dt.strftime("%Y-%m").tolist(),
        "solar": monthly["solar_gen_MW"].round(1).tolist(),
        "wind": monthly["wind_gen_MW"].round(1).tolist(),
        "total_renewable": monthly["total_renewable"].round(1).tolist(),
        "demand": monthly["National Hourly Demand"].round(1).tolist(),
        "renewable_share": monthly["renewable_share"].tolist(),
    }


def scenario_projections(base_demand_mw: float) -> list:
    """Project demand growth under 3 scenarios (2024–2035)."""
    scenarios = [
        {"name": "Conservative (4%)", "rate": 0.04, "color": "#10b981"},
        {"name": "Base Case (6.5%)", "rate": 0.065, "color": "#6366f1"},
        {"name": "High Growth (9%)", "rate": 0.09, "color": "#f59e0b"},
    ]
    years = list(range(2024, 2036))
    result = []
    for s in scenarios:
        demand = [round(base_demand_mw * ((1 + s["rate"]) ** (yr - 2024))) for yr in years]
        result.append({"name": s["name"], "rate": s["rate"], "color": s["color"], "years": years, "demand": demand})
    return result


def load_duration_curve(demand_series: pd.Series) -> dict:
    """Return load duration curve data (200 sample points)."""
    sorted_d = np.sort(demand_series.values)[::-1]
    n = len(sorted_d)
    idx = np.linspace(0, n - 1, 200, dtype=int)
    return {
        "percentile": np.linspace(0, 100, 200).round(2).tolist(),
        "demand": sorted_d[idx].round(0).tolist(),
    }


def peak_heatmap(df: pd.DataFrame) -> dict:
    """Average demand by hour × day_of_week."""
    pivot = df.groupby(["hour", "dayofweek"])["National Hourly Demand"].mean().reset_index()
    pivot.columns = ["hour", "day", "avg_demand"]
    return {
        "hour": pivot["hour"].tolist(),
        "day": pivot["day"].tolist(),
        "demand": pivot["avg_demand"].round(0).tolist(),
    }
