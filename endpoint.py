from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return " Flask Backend is Running!"

# Define /predict endpoint (dummy version)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptoms = data.get("symptoms", [])
    # For now, return dummy response
    return jsonify({
        "received_symptoms": symptoms,
        "prediction": "dummy_disease"
    })

if __name__ == "__main__":
    app.run(debug=True)
