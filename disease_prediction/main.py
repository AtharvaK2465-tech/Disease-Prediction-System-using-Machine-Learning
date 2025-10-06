# main.py
import sys
from joblib import dump
from config import CONFIG
from utils import load_data, quick_review
from cleaning import clean_data
from eda import run_eda
from preprocessing import build_preprocessing_pipeline
from models import baseline_and_compare
from tuning import tune_model
import os

def run_full_pipeline():
    df = load_data(CONFIG["DATA_PATH"])
    if CONFIG["TARGET_COL"] not in df.columns:
        print(f"[ERROR] Target col '{CONFIG['TARGET_COL']}' not found.")
        sys.exit(1)

    df = clean_data(df)
    quick_review(df)
    run_eda(df)

    preprocessor, _, _ = build_preprocessing_pipeline(df)
    best_model_name, best_model, (X_train, X_test, y_train, y_test) = baseline_and_compare(df, CONFIG["TARGET_COL"], preprocessor)

    tuned_model = tune_model(best_model_name, best_model.named_steps["clf"], preprocessor, X_train, y_train)
    dump(tuned_model, os.path.join(CONFIG["MODEL_DIR"], "best_tuned_model.joblib"))
    print("[FINAL] Tuned model saved as best_tuned_model.joblib")


if __name__ == "__main__":
    run_full_pipeline()
