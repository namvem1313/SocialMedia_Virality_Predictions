import pandas as pd

def load_raw_data(path):
    return pd.read_csv(path)

def save_processed_data(df, path):
    df.to_csv(path, index=False)