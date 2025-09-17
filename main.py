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

<<<<<<< HEAD
print(" Training & Testing Datasets Loaded Successfully!\n")
=======
print("Training & Testing Datasets Loaded Successfully!\n")
>>>>>>> 1fe0179 (baseline)

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

<<<<<<< HEAD
print("\n Extra Datasets Loaded Successfully!\n")
=======
print("\nExtra Datasets Loaded Successfully!\n")
>>>>>>> 1fe0179 (baseline)

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

<<<<<<< HEAD
print("\n Cleaned All Datasets!\n")
=======
print("\nCleaned All Datasets!\n")
>>>>>>> 1fe0179 (baseline)

# Save cleaned versions
df_train.to_csv("Training_cleaned.csv", index=False)
df_test.to_csv("Testing_cleaned.csv", index=False)
df_main.to_csv("dataset_cleaned.csv", index=False)
df_desc.to_csv("symptom_Description_cleaned.csv", index=False)
df_precaution.to_csv("symptom_precaution_cleaned.csv", index=False)
df_severity.to_csv("Symptom-severity_cleaned.csv", index=False)

print("All cleaned datasets saved!\n")

# ======================
# 4. Split Features & Target
# ======================
target_column = "prognosis"
if target_column not in df_train.columns:
<<<<<<< HEAD
    raise ValueError(f" Target column '{target_column}' not found! Available columns: {df_train.columns.tolist()}")
=======
    raise ValueError(f"Target column '{target_column}' not found! Available columns: {df_train.columns.tolist()}")
>>>>>>> 1fe0179 (baseline)

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

<<<<<<< HEAD
print(" Feature Scaling Completed!\n")
print(" Final Shapes after Cleaning, Saving & Scaling:")
=======
print("Feature Scaling Completed!\n")
print("Final Shapes after Cleaning, Saving & Scaling:")
>>>>>>> 1fe0179 (baseline)
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
# 7. Encode Categorical Columns
# ======================
le = LabelEncoder()

# Encode Training & Testing target
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Save mapping
disease_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
<<<<<<< HEAD
print(" Label Encoded Target Mapping:\n", disease_mapping, "\n")
=======
print("Label Encoded Target Mapping:\n", disease_mapping, "\n")
>>>>>>> 1fe0179 (baseline)

# Encode df_main (Disease + Symptoms)
df_main_encoded = df_main.copy()
symptom_cols_main = [c for c in df_main.columns if c != "Disease"]

# One-hot encode symptom columns
df_main_encoded = pd.get_dummies(df_main_encoded, columns=symptom_cols_main)

# Label encode disease
df_main_encoded["Disease"] = le.fit_transform(df_main["Disease"])

# ======================
# 8. Severity Mapping
# ======================
severity_map = dict(zip(df_severity["Symptom"], df_severity["weight"]))

df_train_severity = df_train.copy()
for symptom in symptom_cols:
    if symptom in severity_map:
        df_train_severity[symptom] = df_train_severity[symptom] * severity_map.get(symptom, 0)

df_test_severity = df_test.copy()
for symptom in symptom_cols:
    if symptom in severity_map:
        df_test_severity[symptom] = df_test_severity[symptom] * severity_map.get(symptom, 0)

# ======================
# 9. Save Encoded Versions
# ======================
pd.DataFrame(X_train_scaled).to_csv("X_train_scaled.csv", index=False)
pd.DataFrame(X_test_scaled).to_csv("X_test_scaled.csv", index=False)
pd.DataFrame(y_train_encoded).to_csv("y_train_encoded.csv", index=False)
pd.DataFrame(y_test_encoded).to_csv("y_test_encoded.csv", index=False)

df_main_encoded.to_csv("dataset_encoded.csv", index=False)
df_train_severity.to_csv("Training_severity.csv", index=False)
df_test_severity.to_csv("Testing_severity.csv", index=False)

<<<<<<< HEAD
print("\n EDA + Encoding + Severity Mapping Completed Successfully!\n")
=======
print("\nEDA + Encoding + Severity Mapping Completed Successfully!\n")
>>>>>>> 1fe0179 (baseline)
