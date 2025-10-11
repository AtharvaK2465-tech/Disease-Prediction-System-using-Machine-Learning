# app.py
import os
import traceback
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
from flask_mail import Mail, Message
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "outputs", "models", "best_model.joblib")
TRAIN_CSV = os.path.join(BASE_DIR, "Training.csv")

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "Athujojo@2465"
CORS(app) 

# -------------------------------
# Email Configuration
# -------------------------------
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = "atharvkadak1025@gmail.com"        # replace with your email
app.config["MAIL_PASSWORD"] = "oyld uvcv vjhm uwkt"       # use App Password, not your Gmail password!
app.config["MAIL_DEFAULT_SENDER"] = app.config["MAIL_USERNAME"]

mail = Mail(app)

 # harmless when serving templates from same origin; helpful while debugging

# Load training data (for feature names and encoder)
training_data = None
feature_cols = []
encoder = None

if os.path.exists(TRAIN_CSV):
    try:
        training_data = pd.read_csv(TRAIN_CSV)
        if "prognosis" in training_data.columns:
            feature_cols = training_data.drop("prognosis", axis=1).columns.tolist()
            encoder = LabelEncoder()
            encoder.fit(training_data["prognosis"])
            print(f"[INFO] Loaded training CSV. {len(feature_cols)} features found.")
        else:
            feature_cols = training_data.columns.tolist()
            print("[WARN] 'prognosis' column not found in Training.csv — using all columns as features.")
    except Exception:
        print("[ERROR] Failed to read Training.csv")
        traceback.print_exc()
else:
    print(f"[ERROR] Training.csv not found at {TRAIN_CSV}")

# Load model (optional)
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print(f"[INFO] Model loaded from {MODEL_PATH}")
    except Exception:
        print("[ERROR] Failed to load model.joblib")
        traceback.print_exc()
else:
    print(f"[WARN] Model file not found at {MODEL_PATH}. /predict will return an error until the model exists.")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/results")
def results():
    return render_template("results.html")

@app.route("/<page>")
def static_pages(page):
    try:
        return render_template(f"{page}.html")
    except:
        return "Page not found", 404
 
@app.route("/symptoms", methods=["GET"])
def get_symptoms():
    if not feature_cols:
        return jsonify({"error": "No feature columns loaded", "symptoms": []}), 500
    return jsonify({"symptoms": feature_cols})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded on server"}), 500

    try:
        data = request.get_json(force=True) or {}
    except Exception as e:
        return jsonify({"error": f"Invalid JSON: {e}"}), 400

    # ensure all features present; missing -> 0
    input_dict = {}
    for col in feature_cols:
        val = data.get(col, 0)
        # try to coerce to integer 0/1
        try:
            input_dict[col] = int(val)
        except Exception:
            # fallback to 0 if bad value
            input_dict[col] = 0

    input_df = pd.DataFrame([input_dict], columns=feature_cols)

    try:
        pred = model.predict(input_df)
        pred0 = pred[0]

        # If encoder exists and prediction is numeric index, try inverse_transform
        if encoder is not None:
            try:
                # if numeric (int or numpy int64)
                if isinstance(pred0, (int, float)):
                    label = encoder.inverse_transform([int(pred0)])[0]
                else:
                    # if prediction already a label present in encoder.classes_
                    if str(pred0) in encoder.classes_:
                        label = str(pred0)
                    else:
                        # attempt inverse transform (will raise if incompatible)
                        label = encoder.inverse_transform([pred0])[0]
            except Exception:
                # fallback to returned value as string
                label = str(pred0)
        else:
            label = str(pred0)

        return jsonify({"input": input_dict, "prediction": label})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Prediction error: {e}"}), 500
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if model is None:
        return jsonify({"error": "Model not loaded on server"}), 500

    try:
        data = request.get_json(force=True) or {}
    except Exception as e:
        return jsonify({"error": f"Invalid JSON: {e}"}), 400

    # Prepare input (same as /predict)
    input_dict = {}
    for col in feature_cols:
        val = data.get(col, 0)
        try:
            input_dict[col] = int(val)
        except Exception:
            input_dict[col] = 0

    input_df = pd.DataFrame([input_dict], columns=feature_cols)

    try:
        pred = model.predict(input_df)
        pred0 = pred[0]

        if encoder is not None:
            try:
                if isinstance(pred0, (int, float)):
                    label = encoder.inverse_transform([int(pred0)])[0]
                elif str(pred0) in encoder.classes_:
                    label = str(pred0)
                else:
                    label = encoder.inverse_transform([pred0])[0]
            except Exception:
                label = str(pred0)
        else:
            label = str(pred0)

        # return JSON only — for frontend API fetch()
        return jsonify({
            "success": True,
            "prediction": label,
            "input": input_dict
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Prediction error: {e}"}), 500

@app.route("/contact", methods=["POST"])
def contact():
    name = request.form.get("name")
    email = request.form.get("email")
    message = request.form.get("message")

    if not name or not email or not message:
        flash("Please fill out all fields.", "error")
        return redirect(url_for("home"))

    # Email content
    subject = f"New Contact Form Submission from {name}"
    body = f"Name: {name}\nEmail: {email}\n\nMessage:\n{message}"

    try:
        msg = Message(subject=subject, recipients=["atharvkadak1025@gmail.com"], body=body)
        mail.send(msg)
        flash("Thank you! Your message has been sent successfully.", "success")
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")
        flash("Something went wrong while sending your message. Please try again later.", "error")

    return redirect(url_for("home"))

@app.route('/report')
def report():
    return render_template('report.html')


if __name__ == "__main__":
    # run on 0.0.0.0 so `localhost` and ip addresses work; debug=True for tracebacks
    app.run(host="0.0.0.0", port=5000, debug=True)
