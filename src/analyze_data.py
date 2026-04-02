import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH = "enhanced_hourly_electricity_dataset.csv"

def analyze():
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    print(f"Loading {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print("\n--- Project Data Overview ---")
    print(df.head())
    print("\n--- Info ---")
    print(df.info())
    print("\n--- Statistics ---")
    print(df.describe())
    
    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    # Attempt to parse datetime
    # Looking for a column that likely contains 'date' or 'time'
    date_col = None
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            date_col = col
            break
    
    if date_col:
        print(f"\nConverting '{date_col}' to datetime...")
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
        
        # simple plot
        plt.figure(figsize=(15, 5))
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            target_col = numeric_cols[0] # Assume first numeric is target? Or look for 'National Hourly' (from previous context)
            
            # Check for 'National Hourly' specifically from user context
            if 'National Hourly' in df.columns:
                target_col = 'National Hourly'
            
            print(f"Plotting trend for: {target_col}")
            plt.plot(df[date_col], df[target_col], label=target_col, alpha=0.7)
            plt.title(f"{target_col} over Time")
            plt.legend()
            plt.tight_layout()
            plt.savefig("src/data_trend.png")
            print("Saved src/data_trend.png")
        
    else:
        print("Could not identify a datetime column automatically.")

if __name__ == "__main__":
    analyze()
