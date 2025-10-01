# streamlit_app.py
import streamlit as st
import requests

# Flask API URL
API_BASE = "http://127.0.0.1:5000"

st.set_page_config(page_title="Disease Prediction System", layout="wide")

st.title("🩺 Disease Prediction Dashboard")
st.write("Select symptoms and get real-time predictions of possible diseases.")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.info("This dashboard is built with Streamlit and connected to a Flask backend "
            "for real-time ML disease prediction.")

# --- Step 1: Get symptoms list from Flask ---
try:
    resp = requests.get(f"{API_BASE}/symptoms")
    if resp.status_code == 200:
        symptoms_list = resp.json().get("symptoms", [])
    else:
        symptoms_list = []
        st.error("⚠️ Failed to fetch symptoms from API.")
except Exception as e:
    symptoms_list = []
    st.error(f"⚠️ Could not connect to Flask API: {e}")

# --- Step 2: Multi-select for symptoms ---
selected_symptoms = st.multiselect(
    "Choose symptoms you are experiencing:",
    symptoms_list
)

# --- Step 3: Predict button ---
if st.button("🔮 Predict"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        # Build JSON payload
        input_data = {sym: 1 if sym in selected_symptoms else 0 for sym in symptoms_list}

        try:
            resp = requests.post(f"{API_BASE}/predict", json=input_data)
            if resp.status_code == 200:
                result = resp.json().get("prediction")
                st.success(f" Predicted Disease: **{result}**")
            else:
                st.error(f" API Error: {resp.text}")
        except Exception as e:
            st.error(f"Could not connect to Flask API: {e}")
