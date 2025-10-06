# models.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from config import CONFIG
from evaluation import evaluate_model, plot_confusion_matrix

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
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=CONFIG["RANDOM_STATE"]),
        "DecisionTree": DecisionTreeClassifier(random_state=CONFIG["RANDOM_STATE"]),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=CONFIG["RANDOM_STATE"]),
        "SVM": SVC(probability=True, random_state=CONFIG["RANDOM_STATE"]),
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
        plot_confusion_matrix(y_test, y_pred, le.classes_, name)

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

    best_model_name = comp_df.loc[comp_df['cv_mean'].idxmax(), 'model']
    print(f"[AUTO] Best model selected: {best_model_name}")
    best_model = trained_pipelines[best_model_name]
    dump(best_model, os.path.join(CONFIG["MODEL_DIR"], "best_model.joblib"))
    print("[AUTO] Best model saved as best_model.joblib")

    return best_model_name, best_model, (X_train, X_test, y_train, y_test)
