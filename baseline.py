import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# ===============================
# Load dataset
# ===============================
df = pd.read_csv("Training_cleaned.csv")

# Features & Target
X = df.drop(columns=["prognosis"])  # adjust if target column differs
y = df["prognosis"]

print("Original dataset shape:", X.shape)
print("\nClass distribution before cleaning:")
print(y.value_counts())

# ===============================
# Step 1: Remove ultra-rare diseases (<5 samples)
# ===============================
counts = y.value_counts()
rare_classes = counts[counts < 5].index
X = X[~y.isin(rare_classes)]
y = y[~y.isin(rare_classes)]

print("\nClasses kept:", len(y.unique()), "/", len(counts))
print("Samples after rare class removal:", X.shape[0])

# ===============================
# Step 2: Oversample rare classes using SMOTE
# ===============================
smote = SMOTE(random_state=42, k_neighbors=3)  # lowered to handle 5-sample classes
X_res, y_res = smote.fit_resample(X, y)

print("\nDataset shape after SMOTE:", X_res.shape)
print("Balanced class distribution:")
print(pd.Series(y_res).value_counts())

# ===============================
# Step 3: Stratified K-Fold Cross Validation
# ===============================
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

dt_clf = DecisionTreeClassifier(random_state=42)

# Cross-validated accuracy
scores = cross_val_score(dt_clf, X_res, y_res, cv=kf, scoring="accuracy")
print("\nCross-validated accuracy scores:", scores)
print("Mean CV accuracy:", scores.mean())

# Cross-validated predictions
y_pred = cross_val_predict(dt_clf, X_res, y_res, cv=kf)

# ===============================
# Step 4: Evaluation Metrics
# ===============================
print("\n=== Classification Report (Macro) ===")
print(classification_report(y_res, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_res, y_pred))
