import joblib
import pandas as pd
import numpy as np

# Load best model
model_path = "outputs/models/best_model.joblib"   # ✅ adjust path if needed
print(f"Using model file: {model_path}")
model = joblib.load(model_path)

# Load dataset (use Training.csv to get all disease labels)
df = pd.read_csv("Training.csv")

# Get feature columns and target classes
feature_cols = df.columns[:-1]  # all except prognosis
disease_names = df["prognosis"].unique()  # all diseases

def predict_from_symptoms(symptoms, top_n=3):
    # Create input with all 0s
    input_data = {col: 0 for col in feature_cols}
    
    # Mark provided symptoms as 1
    for symptom in symptoms:
        if symptom in input_data:
            input_data[symptom] = 1
    
    # Convert to DataFrame
    X_new = pd.DataFrame([input_data])

    # Predict probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_new)[0]
        # Get top n indices
        top_indices = np.argsort(probs)[::-1][:top_n]
        results = [(disease_names[i], probs[i]*100) for i in top_indices]
        return results
    else:
        # Fallback: just return single prediction
        prediction = model.predict(X_new)[0]
        if isinstance(prediction, (int, float)) and prediction < len(disease_names):
            return [(disease_names[int(prediction)], 100.0)]
        return [(prediction, 100.0)]

# Example usage:
example_symptoms = ["itching", "skin_rash"]
predictions = predict_from_symptoms(example_symptoms, top_n=3)

print("Top Predictions:")
for disease, prob in predictions:
    print(f"- {disease}: {prob:.2f}%")
# -----------------------