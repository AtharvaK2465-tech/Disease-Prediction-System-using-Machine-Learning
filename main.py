# =============================
# Disease Prediction Project
# Task: Load & Review Datasets
# =============================

import pandas as pd

# =============================
# Load Training & Testing datasets
# =============================
train_path = "Training.csv"
test_path = "Testing.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

print("✅ Training & Testing Datasets Loaded Successfully!\n")

# ===== Review Training Dataset =====
print("📌 Training Dataset Review")
print("Shape:", df_train.shape, "\n")
print("Columns:\n", df_train.columns.tolist(), "\n")
print("Data Types:\n", df_train.dtypes, "\n")
print("First 5 rows:\n", df_train.head(), "\n")
print("Missing values:\n", df_train.isnull().sum(), "\n")
print("Summary stats:\n", df_train.describe(), "\n")

# ===== Review Testing Dataset =====
print("\n📌 Testing Dataset Review")
print("Shape:", df_test.shape, "\n")
print("Columns:\n", df_test.columns.tolist(), "\n")
print("Data Types:\n", df_test.dtypes, "\n")
print("First 5 rows:\n", df_test.head(), "\n")
print("Missing values:\n", df_test.isnull().sum(), "\n")
print("Summary stats:\n", df_test.describe(), "\n")

# =============================
# Load Extra Datasets (Uploaded)
# =============================
dataset_path = "dataset.csv"
desc_path = "symptom_Description.csv"
precaution_path = "symptom_precaution.csv"
severity_path = "Symptom-severity.csv"

df_main = pd.read_csv(dataset_path)
df_desc = pd.read_csv(desc_path)
df_precaution = pd.read_csv(precaution_path)
df_severity = pd.read_csv(severity_path)

print("\n✅ Extra Datasets Loaded Successfully!\n")

# ===== Review Main Dataset =====
print("📌 Main Dataset Review (dataset.csv)")
print("Shape:", df_main.shape, "\n")
print("Columns:\n", df_main.columns.tolist(), "\n")
print("First 5 rows:\n", df_main.head(), "\n")
print("Missing values:\n", df_main.isnull().sum(), "\n")

# ===== Review Symptom Description =====
print("\n📌 Symptom Description Dataset Review")
print("Shape:", df_desc.shape, "\n")
print("First 5 rows:\n", df_desc.head(), "\n")

# ===== Review Symptom Precaution =====
print("\n📌 Symptom Precaution Dataset Review")
print("Shape:", df_precaution.shape, "\n")
print("First 5 rows:\n", df_precaution.head(), "\n")

# ===== Review Symptom Severity =====
print("\n📌 Symptom Severity Dataset Review")
print("Shape:", df_severity.shape, "\n")
print("First 5 rows:\n", df_severity.head(), "\n")
