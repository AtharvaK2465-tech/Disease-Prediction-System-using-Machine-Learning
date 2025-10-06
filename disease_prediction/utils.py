# utils.py
import pandas as pd
import numpy as np

def load_data(path: str) -> pd.DataFrame:
    print(f"[INFO] Loading data from: {path}")
    if path.lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    print(f"[INFO] Loaded shape: {df.shape}")
    return df

def quick_review(df: pd.DataFrame) -> None:
    print("\n[REVIEW] Head:\n", df.head())
    print("\n[REVIEW] Info:")
    print(df.info())
    print("\n[REVIEW] Missing values:\n", df.isnull().sum().sort_values(ascending=False).head(20))
