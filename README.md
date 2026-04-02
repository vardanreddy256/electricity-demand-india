# ⚡ India Power Grid Analytics: Multi-Granularity Electricity Forecasting

## Overview
India’s power sector is undergoing a paradigm shift driven by rapid economic growth and urbanization. Legacy forecasting methods are insufficient for managing dynamic, non-linear complexities caused by weather and intermittent renewables. 

This project provides a **data-driven, high-frequency machine learning forecasting system** that predicts electricity demand across the Indian grid. The system uses an ensemble of `XGBoost` and `RandomForest` models to deliver predictions across 7 granularities (from second-level synthesis to year-level aggregation).

## ✨ Key Features
- **Multi-Granularity Engine**: Forecasts demand across 7 time levels: **Second**, **Minute**, **Hour**, **Day**, **Week**, **Month**, **Year**.
- **State-of-the-art ML Models**: Uses an automated competition between `XGBoost` (with 40-round early stopping) and `Random Forest` regressors.
- **Rich Feature Engineering**: Leverages 35+ engineered features including cyclical temporal encoding (sin/cos), weather data, and lag/rolling statistical features up to 168 hours.
- **Interactive Dashboard**: A premium, dark-themed Streamlit UI featuring interactive Plotly charts, dynamic confidence intervals, regional heatmaps, API integrations, and CSV forecast exports.
- **High Accuracy (Grade S)**: Achieved >95% R² scores and <2% MAPE across 6 national and regional power grid zones.

## 🚀 Model Performance Metrics

| Region | Best Model | MAPE (%) | R² Score | RMSE (MW) |
|--------|-----------|----------|---------|-----------|
| **National Hourly Demand** | XGBoost | 1.36% | 0.9513 | ~4315 |
| **Northern Region** | XGBoost | 1.57% | 0.9874 | ~1180 |
| **Western Region** | XGBoost | 1.19% | 0.9716 | ~1012 |
| **Eastern Region** | RandomForest | 1.54% | 0.9766 | ~499 |
| **Southern Region** | XGBoost | 1.53% | 0.9726 | ~1186 |
| **North-Eastern Region** | XGBoost | 1.88% | 0.9830 | ~58 |

## 📦 Tech Stack
- **Data Manipulation & Math**: `pandas`, `numpy`, `scipy` (Cubic Splines)
- **Machine Learning**: `scikit-learn`, `xgboost` 
- **Web App / UI**: `streamlit`, `plotly`
- **Deployment**: `Git LFS`, `Render` 

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vardanreddy256/electricity-demand-india.git
   cd electricity-demand-india
   ```

2. **Setup Git LFS (Large File Storage) to download models and datasets:**
   ```bash
   git lfs install
   git lfs pull
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Dashboard locally:**
   ```bash
   streamlit run app.py
   ```

*(Optional) To retrain the ML pipeline from scratch, run `python src/train_multi_granularity.py`.*

## ☁️ Deployment
This application is fully pre-configured for free-tier deployment on **Render** using a Blueprint config. 
- In Render Dashboard, click **New > Blueprint**.
- Connect the Git repository.
- Ensure the start command is configured correctly in settings: `streamlit run app.py --server.port $PORT`.

---
*Developed by Vardan Reddy*
