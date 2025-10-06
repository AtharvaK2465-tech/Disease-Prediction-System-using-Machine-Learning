# tuning.py
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from config import CONFIG

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
