import joblib
import pandas as pd
from flask import Flask, request, jsonify , render_template
from sklearn.preprocessing import LabelEncoder

# Load best model (pipeline with preprocessing inside)
model = joblib.load("outputs/models/best_model.joblib")

# Load training data once to know feature columns
training_data = pd.read_csv("Training.csv")
feature_cols = training_data.drop("prognosis", axis=1).columns.tolist()

# Recreate LabelEncoder (same order as during training)
encoder = LabelEncoder()
encoder.fit(training_data["prognosis"])

app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Flask Backend is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Example input: {"fever": 1, "cough": 0, "nausea": 1}

    # Build dataframe with same columns as training
    input_df = pd.DataFrame([data], columns=feature_cols)

    try:
        prediction_encoded = model.predict(input_df)[0]
        prediction_label = encoder.inverse_transform([prediction_encoded])[0]

        return jsonify({
            "input": data,
            "prediction": prediction_label
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
