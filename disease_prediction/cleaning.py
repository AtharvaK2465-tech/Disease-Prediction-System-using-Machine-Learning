# cleaning.py
import numpy as np
import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("[CLEAN] Dropping fully empty columns...")
    df = df.dropna(axis=1, how="all")
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "missing")
    print("[CLEAN] After cleaning, shape:", df.shape)
    return df
