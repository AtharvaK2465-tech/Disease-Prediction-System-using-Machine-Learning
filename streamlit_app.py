import streamlit as st
import requests
import pandas as pd

API_BASE = "http://127.0.0.1:5000"

st.set_page_config(page_title="Disease Prediction System", layout="wide")

# Title
st.markdown("<h1 style='text-align:center; color:#4CAF50;'>🩺 Disease Prediction Dashboard</h1>", unsafe_allow_html=True)
st.write("Select symptoms and get real-time predictions of possible diseases.")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966484.png", width=100)
    st.header(" About")
    st.info("This dashboard is built with **Streamlit** and connected to a **Flask backend** "
            "for real-time ML disease prediction.")
    st.markdown("---")
    st.markdown(" *Developed by G10*")

# --- Step 1: Get symptoms list from Flask ---
try:
    resp = requests.get(f"{API_BASE}/symptoms")
    if resp.status_code == 200:
        symptoms_list = resp.json().get("symptoms", [])
    else:
        symptoms_list = []
        st.error(" Failed to fetch symptoms from API.")
except Exception as e:
    symptoms_list = []
    st.error(f" Could not connect to Flask API: {e}")

# --- Step 2: Multi-select for symptoms ---
selected_symptoms = st.multiselect(
    " Choose symptoms you are experiencing:",
    symptoms_list,
    help="You can select multiple symptoms"
)

st.caption(f"You have selected **{len(selected_symptoms)}** symptoms out of {len(symptoms_list)} available.")

# --- Step 3: Predict button ---
if st.button(" Predict"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        input_data = {sym: 1 if sym in selected_symptoms else 0 for sym in symptoms_list}
        try:
            with st.spinner("Analyzing your symptoms... "):
                resp = requests.post(f"{API_BASE}/predict", json=input_data)

            if resp.status_code == 200:
                result = resp.json().get("prediction")
                st.success(f" Predicted Disease: **{result}**")

                # Example: Probability visualization if API supports it
                probs = resp.json().get("probabilities", None)
                if probs:
                    df = pd.DataFrame(list(probs.items()), columns=["Disease", "Probability"])
                    st.bar_chart(df.set_index("Disease"))

            else:
                st.error(f" API Error: {resp.text}")
        except Exception as e:
            st.error(f"Could not connect to Flask API: {e}")

# --- Step 4: Extra Info Cards ---
st.markdown("---")
st.subheader(" Health Tips & Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="background-color:#f8f9fa; padding:20px; border-radius:10px; box-shadow:2px 2px 8px #ddd;">
    <h4 style="color:#2E86C1;">🥦 Nutrition</h4>
    <p>Eat a balanced diet rich in fruits, vegetables, lean proteins, and whole grains.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background-color:#f8f9fa; padding:20px; border-radius:10px; box-shadow:2px 2px 8px #ddd;">
    <h4 style="color:#27AE60;">🏋️ Exercise</h4>
    <p>Engage in at least 30 minutes of moderate physical activity daily.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background-color:#f8f9fa; padding:20px; border-radius:10px; box-shadow:2px 2px 8px #ddd;">
    <h4 style="color:#E67E22;">💧 Hydration</h4>
    <p>Drink 8–10 glasses of water every day to stay hydrated and energized.</p>
    </div>
    """, unsafe_allow_html=True)

# Another row of cards
col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("""
    <div style="background-color:#f8f9fa; padding:20px; border-radius:10px; box-shadow:2px 2px 8px #ddd;">
    <h4 style="color:#8E44AD;">🛌 Rest</h4>
    <p>Get 7–8 hours of sleep daily for optimal health and recovery.</p>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div style="background-color:#f8f9fa; padding:20px; border-radius:10px; box-shadow:2px 2px 8px #ddd;">
    <h4 style="color:#C0392B;">🚭 Lifestyle</h4>
    <p>Avoid smoking, limit alcohol, and manage stress for a healthier life.</p>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown("""
    <div style="background-color:#f8f9fa; padding:20px; border-radius:10px; box-shadow:2px 2px 8px #ddd;">
    <h4 style="color:#F39C12;">🧘 Mental Health</h4>
    <p>Practice mindfulness, meditation, or hobbies to keep stress under control.</p>
    </div>
    """, unsafe_allow_html=True)
