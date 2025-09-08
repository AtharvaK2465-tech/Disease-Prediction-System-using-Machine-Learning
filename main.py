# =============================
# Disease Prediction Project
# Task: Load & Review Dataset
# =============================

import pandas as pd

# Load Training and Testing datasets
train_path = "Training.csv"
test_path = "Testing.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

print("✅ Datasets Loaded Successfully!\n")

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
