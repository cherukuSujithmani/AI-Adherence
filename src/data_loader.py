import pandas as pd

def load_data(path):
    try:
        df = pd.read_csv(path)
        print("✅ Data loaded successfully")
        print("Shape:", df.shape)
        return df
    except Exception as e:
        print("❌ Error loading data:", e)
        return None