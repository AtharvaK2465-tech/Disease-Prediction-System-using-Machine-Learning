from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Base route
@app.route("/")
def home():
    return {"message": "Flask backend is running successfully!"}

# Predict route (dummy response for now)
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Extract symptoms from request
    symptoms = data.get("symptoms", [])

    # Dummy prediction response
    response = {
        "input_symptoms": symptoms,
        "predicted_disease": "Dummy Disease",
        "confidence": "N/A (dummy response)"
    }
    return jsonify(response)

# Run the server
if __name__ == "__main__":
    app.run(debug=True)
