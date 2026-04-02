import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
import joblib
import os

DATA_PATH = "enhanced_hourly_electricity_dataset.csv"
MODEL_PATH = "model_dict.pkl"

def create_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    return df

def train():
    print("Loading data...")
    if not os.path.exists(DATA_PATH):
        print("Data file not found!")
        return

    df = pd.read_csv(DATA_PATH)
    
    # Identify datetime column
    date_col = None
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            date_col = col
            break
            
    if not date_col:
        print("Could not find datetime column")
        return

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.set_index(date_col)
    df = df.sort_index()

    # Identify Target Columns (National + Regions)
    # We look for columns containing "Hourly" or "Demand" and exclude "National" to find regions
    # But we want National too.
    target_cols = [c for c in df.columns if 'Hourly' in c or 'Demand' in c]
    # Filter out non-numeric if any
    target_cols = [c for c in target_cols if pd.api.types.is_numeric_dtype(df[c])]
    
    print(f"Targets identified: {target_cols}")

    # Feature Engineering
    df = create_features(df)
    
    # Encode Season if exists
    if 'season' in df.columns:
        le = LabelEncoder()
        df['season_encoded'] = le.fit_transform(df['season'].astype(str))
        FEATURES = ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'season_encoded']
        # Also include weather if available
        weather_cols = ['temperature_C', 'humidity_percent', 'heat_index_C', 'solar_gen_MW', 'wind_gen_MW']
        for wc in weather_cols:
            if wc in df.columns:
                FEATURES.append(wc)
    else:
        FEATURES = ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear']

    print(f"Features: {FEATURES}")

    model_dict = {}
    
    for target in target_cols:
        print(f"\nTraining for: {target}")
        
        # Remove rows with NaN in target
        temp_df = df.dropna(subset=[target] + FEATURES)
        
        # Split
        split_idx = int(len(temp_df) * 0.8)
        train_df = temp_df.iloc[:split_idx]
        test_df = temp_df.iloc[split_idx:]

        X_train = train_df[FEATURES]
        y_train = train_df[target]
        X_test = test_df[FEATURES]
        y_test = test_df[target]

        reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                               n_estimators=500,
                               early_stopping_rounds=20,
                               objective='reg:squarederror',
                               max_depth=3,
                               learning_rate=0.05)
        
        reg.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=False)
        
        preds = reg.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, preds)
        print(f"MAPE ({target}): {mape:.4f}")
        
        model_dict[target] = reg

    # Save dictionary of models
    print(f"Saving all models to {MODEL_PATH}")
    joblib.dump(model_dict, MODEL_PATH)
    
    # Save Feature columns context
    joblib.dump(FEATURES, "features.pkl")
    if 'season' in df.columns:
        joblib.dump(le, "season_encoder.pkl")

if __name__ == "__main__":
    train()
