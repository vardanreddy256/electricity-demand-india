# ⚡ ElectricAI India

> **Electricity Demand Forecasting & Grid Planning Dashboard for India**  
> Full-stack data science project | FastAPI + ML (XGBoost, LightGBM, RandomForest) + Premium Web Dashboard

## 👥 Team
- **[Your Name]** — Role/Contribution
- **[Teammate 1 Name]** — Role/Contribution
- **[Teammate 2 Name]** — Role/Contribution

---

## 🖥️ Dashboard Preview

| Overview | Regional Analysis |
|---|---|
| National KPIs, demand trends, heatmaps | 5-region breakdown, donut chart, monthly comparison |

| Forecasting | Grid Planning | Model Comparison |
|---|---|---|
| Actual vs Predicted with CI | Capacity gap, renewables, scenario sliders | Radar chart, metrics table |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run (auto-trains models on first launch)
```bash
python run.py
```

### 3. Open dashboard
- **Dashboard** → http://localhost:8000  
- **API Docs** → http://localhost:8000/docs

> ℹ️ First run trains 3 ML models (~5–10 min). Subsequent runs start instantly.

---

## 📁 Project Structure

```
electricity-demand-india/
│
├── 📂 data/
│   ├── raw/                         ← Source CSV dataset (46,728 rows)
│   └── processed/                   ← cleaned.parquet + features.parquet
│
├── 📂 src/                          ← ML pipeline (pure Python, no notebooks)
│   ├── preprocessing.py             ← Data loading, cleaning, validation
│   ├── feature_engineering.py       ← Lag features, cyclical encoding, rolling stats
│   ├── evaluator.py                 ← MAE, RMSE, MAPE, R² metrics
│   ├── grid_planning.py             ← Capacity gap, renewable & scenario analysis
│   └── models/
│       ├── base.py                  ← Abstract model interface
│       ├── xgboost_model.py         ← XGBoost with early stopping
│       ├── lightgbm_model.py        ← LightGBM
│       └── randomforest_model.py    ← Random Forest
│
├── 📂 scripts/
│   └── run_pipeline.py              ← End-to-end training script
│
├── 📂 saved_models/                 ← Trained .pkl files + metrics.json
│
├── 📂 api/                          ← FastAPI backend
│   ├── main.py                      ← App entry point + static file serving
│   ├── dependencies.py              ← Shared cached data access
│   └── routers/
│       ├── overview.py              ← /api/overview/* (KPIs, trends, heatmap)
│       ├── regional.py              ← /api/regional/* (breakdown, stats, compare)
│       ├── forecast.py              ← /api/forecast/* (predict, feature importance)
│       ├── grid.py                  ← /api/grid/* (capacity, renewables, scenarios)
│       └── compare.py               ← /api/compare/* (metrics, radar, predictions)
│
├── 📂 frontend/                     ← Premium dark-themed SPA
│   ├── index.html                   ← Main HTML shell
│   ├── css/
│   │   ├── main.css                 ← Design system (tokens, layout, charts)
│   │   └── components.css           ← Reusable UI components
│   └── js/
│       ├── utils.js                 ← Helpers: fmt, animateCounter, chartOpts
│       ├── api.js                   ← All fetch wrappers for backend endpoints
│       ├── app.js                   ← SPA router, sidebar, navigation
│       └── pages/
│           ├── overview.js          ← Overview page logic
│           ├── regional.js          ← Regional analysis logic
│           ├── forecast.js          ← Forecasting page logic
│           ├── grid.js              ← Grid planning logic
│           └── compare.js           ← Model comparison logic
│
├── run.py                           ← One-command launcher
└── requirements.txt
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | `data/raw/enhanced_hourly_electricity_dataset.csv` |
| Records | **46,728 hourly rows** |
| Date Range | Jan 2019 – Apr 2024 |
| Regions | Northern, Western, Eastern, Southern, North-Eastern |
| Features | National demand, regional demand, temperature, humidity, solar, wind, holidays |

---

## 🤖 Model Performance

| Model | MAE (MW) | MAPE (%) | R² Score | Train Time |
|---|---|---|---|---|
| XGBoost | 3,502 | 1.82% | 0.9406 | ~220s |
| LightGBM | 3,411 | 1.77% | 0.9401 | ~21s ⚡ |
| **RandomForest** ⭐ | **3,317** | **1.77%** | **0.9516** | ~80s |

> RandomForest achieves the best R² (0.9516), accurately tracking India's complex seasonal and daily demand cycles.

### Features Used (37 total)
- **Cyclical**: hour sin/cos, day-of-week sin/cos, month sin/cos
- **Lag**: 1h, 2h, 3h, 24h, 48h, 168h (1 week) demand lags
- **Rolling**: 6h, 24h, 168h mean and std
- **Calendar**: is_weekend, season, is_holiday
- **Weather**: temperature, humidity
- **Renewable**: solar_MW, wind_MW, renewable_penetration_ratio

---

## 🛠️ API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/overview/kpis` | Peak, average, load factor, renewable share |
| GET | `/api/overview/trend` | National demand time series (filterable) |
| GET | `/api/overview/heatmap` | Hour × day-of-week demand matrix |
| GET | `/api/overview/yearly-stats` | Year-by-year average and peak demand |
| GET | `/api/regional/breakdown` | 5-region demand share (last 24h avg) |
| GET | `/api/regional/trend` | Weekly trend for selected region |
| GET | `/api/regional/compare` | Monthly all-region comparison |
| GET | `/api/regional/stats` | Avg, peak, min per region |
| GET | `/api/forecast/predict` | Model predictions + confidence intervals |
| GET | `/api/forecast/actuals` | Historical actual demand |
| GET | `/api/forecast/feature-importance` | Top feature importances per model |
| GET | `/api/grid/capacity-summary` | Installed capacity by type |
| GET | `/api/grid/capacity-gap` | Peak demand vs installed capacity |
| GET | `/api/grid/renewables` | Solar + wind monthly generation |
| GET | `/api/grid/load-curve` | Load duration curve (percentiles) |
| GET | `/api/grid/scenarios` | 2024–2035 growth scenarios (parametric) |
| GET | `/api/grid/seasonal-analysis` | Seasonal avg/peak/min demand |
| GET | `/api/compare/metrics` | All model metrics + best model |
| GET | `/api/compare/predictions` | Side-by-side predictions (all models) |
| GET | `/api/compare/radar` | Normalized radar chart data |
| GET | `/api/health` | API health check |
| GET | `/docs` | Interactive Swagger UI |

---

## 💡 Dashboard Features

### 📊 Overview
- Animated KPI counters (Peak Demand, Avg, Load Factor, Renewable Share, YoY Growth)
- Time-period filter buttons: 1M / 3M / 6M / 1Y / 2Y / ALL
- National demand trend with indigo gradient fill
- Yearly demand bar chart (avg + peak)
- Hour × weekday demand heatmap

### 🗺️ Regional Analysis
- Donut chart with 5 India grid regions
- Interactive stats table with color-coded chips
- Region selector for monthly trend
- Multi-region monthly comparison chart

### 📈 Forecasting
- Dropdown for XGBoost / LightGBM / RandomForest
- Forecast horizon: 24h → 720h (1 month)
- Actual vs Predicted plot with confidence bands
- Feature importance horizontal bar chart
- Prediction error distribution histogram

### ⚡ Grid Planning
- 5 capacity KPI cards
- Installed capacity breakdown (Thermal, Hydro, Solar, Wind, Nuclear)
- Capacity gap: demand vs installed capacity timeline
- Renewable generation trend (Solar + Wind stacked)
- Load duration curve
- **Interactive scenario sliders** — adjust Conservative/Base/High growth rates, see 2035 projections update live

### 🤖 Model Comparison
- Best model trophy banner
- Side-by-side metrics table
- Radar chart (MAE, RMSE, MAPE, R² — normalized)
- All-models overlay chart (168h comparison)
- MAE and R² bar charts

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| ML Models | XGBoost, LightGBM, scikit-learn (RandomForest) |
| Data Pipeline | pandas, numpy |
| Model Persistence | joblib |
| Frontend | HTML5 + CSS3 (Vanilla, no framework) |
| Charts | Chart.js 4 |
| Fonts | Inter (Google Fonts) |

---

## 🔄 Re-training Models

```bash
python scripts/run_pipeline.py
```

Models are saved to `saved_models/` as `.pkl` files with `metrics.json`.

---

## 📝 License

MIT License — built for educational and research purposes.
# electricity-demand-india
