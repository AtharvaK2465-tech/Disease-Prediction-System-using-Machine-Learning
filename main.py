import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ======================
# 1. Load Training & Testing Datasets
# ======================
train_path = "Training.csv"
test_path = "Testing.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

print("✅ Training & Testing Datasets Loaded Successfully!\n")
print("📌 Training Dataset Review")
print("Shape:", df_train.shape, "\n")
print("Columns:\n", df_train.columns.tolist(), "\n")
print("Data Types:\n", df_train.dtypes, "\n")
print("First 5 rows:\n", df_train.head(), "\n")
print("Missing values:\n", df_train.isnull().sum(), "\n")

print("\n📌 Testing Dataset Review")
print("Shape:", df_test.shape, "\n")
print("Columns:\n", df_test.columns.tolist(), "\n")
print("Data Types:\n", df_test.dtypes, "\n")
print("First 5 rows:\n", df_test.head(), "\n")
print("Missing values:\n", df_test.isnull().sum(), "\n")

# ======================
# 2. Load Extra Datasets
# ======================
dataset_path = "dataset.csv"
desc_path = "symptom_Description.csv"
precaution_path = "symptom_precaution.csv"
severity_path = "Symptom-severity.csv"

df_main = pd.read_csv(dataset_path)
df_desc = pd.read_csv(desc_path)
df_precaution = pd.read_csv(precaution_path)
df_severity = pd.read_csv(severity_path)

print("\n✅ Extra Datasets Loaded Successfully!\n")
print("📌 Main Dataset Review (dataset.csv)")
print("Shape:", df_main.shape, "\n")
print("Columns:\n", df_main.columns.tolist(), "\n")
print("First 5 rows:\n", df_main.head(), "\n")

# ======================
# 3. Clean Function
# ======================
def clean_dataset(df):
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df.drop_duplicates()
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df

# Clean all
df_train = clean_dataset(df_train)
df_test = clean_dataset(df_test)
df_main = clean_dataset(df_main)
df_desc = clean_dataset(df_desc)
df_precaution = clean_dataset(df_precaution)
df_severity = clean_dataset(df_severity)

print("\n✅ Cleaned All Datasets!\n")

# Save cleaned versions
df_train.to_csv("Training_cleaned.csv", index=False)
df_test.to_csv("Testing_cleaned.csv", index=False)
df_main.to_csv("dataset_cleaned.csv", index=False)
df_desc.to_csv("symptom_Description_cleaned.csv", index=False)
df_precaution.to_csv("symptom_precaution_cleaned.csv", index=False)
df_severity.to_csv("Symptom-severity_cleaned.csv", index=False)

print("💾 All cleaned datasets saved!\n")

# ======================
# 4. Split Features & Target
# ======================
target_column = "prognosis"
if target_column not in df_train.columns:
    raise ValueError(f"❌ Target column '{target_column}' not found! Available columns: {df_train.columns.tolist()}")

X_train = df_train.drop(columns=[target_column])
y_train = df_train[target_column]
X_test = df_test.drop(columns=[target_column])
y_test = df_test[target_column]

# ======================
# 5. Scaling
# ======================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Feature Scaling Completed!\n")
print("📌 Final Shapes after Cleaning, Saving & Scaling:")
print("X_train:", X_train_scaled.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test_scaled.shape)
print("y_test:", y_test.shape)

# ======================
# 6. EDA (Symptom Frequency)
# ======================
symptom_cols = [col for col in df_train.columns if col != target_column]

symptom_counts = df_train[symptom_cols].sum().sort_values(ascending=False)
plt.figure(figsize=(12,6))
sns.barplot(x=symptom_counts.values, y=symptom_counts.index)
plt.title("Symptom Frequency in Training Dataset")
plt.xlabel("Count")
plt.ylabel("Symptom")
plt.show()

# ======================
# 7. Encode Categorical for Correlation Heatmap
# ======================
df_encoded = df_main.copy()
le = LabelEncoder()

for col in df_encoded.columns:
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

plt.figure(figsize=(12,8))
sns.heatmap(df_encoded.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap (Label Encoded Features)")
plt.show()

print("\n✅ EDA + Encoding Completed Successfully!\n")
