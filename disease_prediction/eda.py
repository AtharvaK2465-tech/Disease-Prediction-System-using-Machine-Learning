# eda.py
import os
import matplotlib.pyplot as plt
import seaborn as sns
from config import CONFIG

sns.set(style="whitegrid")

def run_eda(df):
    print("[EDA] Running exploratory data analysis...")

    # Target distribution
    plt.figure(figsize=(10, 5))
    sns.countplot(y=df[CONFIG["TARGET_COL"]], order=df[CONFIG["TARGET_COL"]].value_counts().index)
    plt.title("Target Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["EDA_DIR"], "target_distribution.png"))
    plt.close()

    # Correlation heatmap for numeric features
    num_df = df.select_dtypes(include=["int64", "float64"])
    if not num_df.empty:
        plt.figure(figsize=(12, 10))
        sns.heatmap(num_df.corr(), cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG["EDA_DIR"], "correlation_heatmap.png"))
        plt.close()

    print("[EDA] Plots saved to:", CONFIG["EDA_DIR"])
