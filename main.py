import os
import sys
import warnings
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from joblib import dump

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# -----------------------
# CONFIG
# -----------------------
CONFIG = {
    "DATA_PATH": "Training.csv",
    "TARGET_COL": "prognosis",
    "TEST_SIZE": 0.2,
    "RANDOM_STATE": 42,
    "OUTPUT_DIR": "outputs",
    "EDA_DIR": "outputs/eda",
    "MODEL_DIR": "outputs/models",
    "PRED_DIR": "outputs/predictions",
    "CV_FOLDS": 5,
}
# -----------------------

# Create output folders
for d in (CONFIG["OUTPUT_DIR"], CONFIG["EDA_DIR"], CONFIG["MODEL_DIR"], CONFIG["PRED_DIR"]):
    os.makedirs(d, exist_ok=True)


# -----------------------
# Utils
# -----------------------
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


# -----------------------
# Data Cleaning
# -----------------------
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


# -----------------------
# EDA
# -----------------------
def run_eda(df: pd.DataFrame):
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


# -----------------------
# Preprocessing pipeline
# -----------------------
def build_preprocessing_pipeline(df):
    feature_df = df.drop(CONFIG["TARGET_COL"], axis=1, errors="ignore")
    num_cols = feature_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = feature_df.select_dtypes(include=["object"]).columns.tolist()

    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
        ]), cat_cols))

    return ColumnTransformer(transformers=transformers), num_cols, cat_cols


# -----------------------
# Model Evaluation
# -----------------------
def evaluate_model(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
        except Exception:
            metrics["roc_auc"] = np.nan
    return metrics


# -----------------------
# Baseline & Comparison
# -----------------------
def baseline_and_compare(df, target, preprocessor):
    X = df.drop(columns=[target])
    y = df[target]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=CONFIG["TEST_SIZE"],
        random_state=CONFIG["RANDOM_STATE"], stratify=y_encoded
    )

    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=CONFIG["RANDOM_STATE"]),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=CONFIG["RANDOM_STATE"]),
        "SVM": SVC(probability=True, random_state=CONFIG["RANDOM_STATE"]),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=CONFIG["RANDOM_STATE"]),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=CONFIG["RANDOM_STATE"]),
    }

    results, trained_pipelines = {}, {}

    for name, model in models.items():
        print(f"\n[MODEL] Training: {name}")
        pipe = Pipeline([("preprocessor", preprocessor), ("clf", model)])
        try:
            cv = KFold(n_splits=CONFIG["CV_FOLDS"], shuffle=True, random_state=CONFIG["RANDOM_STATE"])
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
        except Exception as e:
            print(f"[WARN] CV failed for {name}: {e}")
            cv_scores = [np.nan]

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        try:
            y_proba = pipe.predict_proba(X_test)
        except Exception:
            y_proba = None

        metrics = evaluate_model(y_test, y_pred, y_proba)
        print(f"[RESULT] {name}: {metrics}")

        results[name] = {"cv_mean": np.nanmean(cv_scores), "cv_std": np.nanstd(cv_scores), "test_metrics": metrics}
        trained_pipelines[name] = pipe
        dump(pipe, os.path.join(CONFIG["MODEL_DIR"], f"{name}.joblib"))

    pd.Series(class_mapping).to_csv(os.path.join(CONFIG["OUTPUT_DIR"], "class_mapping.csv"))

    comp_df = pd.DataFrame([
        {"model": k, "cv_mean": v["cv_mean"], "cv_std": v["cv_std"], **v["test_metrics"]}
        for k, v in results.items()
    ]).sort_values("cv_mean", ascending=False)

    comp_df.to_csv(os.path.join(CONFIG["OUTPUT_DIR"], "model_comparison.csv"), index=False)
    print("[INFO] Model comparison saved.")

    # Auto-pick best
    best_model_name = comp_df.loc[comp_df['cv_mean'].idxmax(), 'model']
    print(f"[AUTO] Best model selected: {best_model_name}")
    best_model = trained_pipelines[best_model_name]
    dump(best_model, os.path.join(CONFIG["MODEL_DIR"], "best_model.joblib"))
    print("[AUTO] Best model saved as best_model.joblib")

    return best_model_name, best_model, (X_train, X_test, y_train, y_test)


# -----------------------
# Hyperparameter Tuning
# -----------------------
def tune_model(model_name, model, preprocessor, X_train, y_train):
    print(f"[TUNE] Hyperparameter tuning for {model_name}...")

    param_grids = {
        "DecisionTree": {"clf__max_depth": [5, 10, 20, None], "clf__min_samples_split": [2, 5, 10]},
        "LogisticRegression": {"clf__C": [0.1, 1, 10], "clf__solver": ["liblinear", "lbfgs"]},
        "SVM": {"clf__C": [0.1, 1, 10], "clf__kernel": ["linear", "rbf"]},
        "RandomForest": {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 10, 20]},
        "XGBoost": {"clf__n_estimators": [100, 200], "clf__max_depth": [3, 6, 10], "clf__learning_rate": [0.01, 0.1, 0.2]},
    }

    param_grid = param_grids.get(model_name, {})
    pipe = Pipeline([("preprocessor", preprocessor), ("clf", model)])

    if param_grid:
        random_search = RandomizedSearchCV(pipe, param_grid, cv=3, n_iter=5,
                                           scoring="accuracy", random_state=CONFIG["RANDOM_STATE"], n_jobs=-1)
        random_search.fit(X_train, y_train)
        print("[TUNE] Best params (RandomizedSearchCV):", random_search.best_params_)

        grid_search = GridSearchCV(pipe, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
        grid_search.fit(X_train, y_train)
        print("[TUNE] Best params (GridSearchCV):", grid_search.best_params_)
        return grid_search.best_estimator_

    else:
        print("[TUNE] No tuning grid defined for this model.")
        return pipe


# -----------------------
# Orchestrator
# -----------------------
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
# -----------------------