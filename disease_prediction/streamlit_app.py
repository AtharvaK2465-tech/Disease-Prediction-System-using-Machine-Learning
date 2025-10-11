# dashboard.py
import streamlit as st
import requests
import pandas as pd

API_BASE = "http://127.0.0.1:5000"

st.set_page_config(page_title="Disease Prediction System", layout="wide")

# Title
st.markdown("<h1 style='text-align:center; color:#4CAF50;'>Disease Prediction Dashboard</h1>", unsafe_allow_html=True)
st.write("Select symptoms and get AI-based disease predictions in real-time.")

# Sidebar Info
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966484.png", width=100)
    st.header("About")
    st.info("This dashboard uses **Streamlit** + **Flask API** for ML-based disease prediction.")
    st.markdown("---")
    st.markdown("Developed by **Group E**")

# Fetch symptoms from Flask
try:
    resp = requests.get(f"{API_BASE}/symptoms")
    if resp.status_code == 200:
        symptoms_list = resp.json().get("symptoms", [])
        st.sidebar.success(f"Loaded {len(symptoms_list)} symptoms ")
    else:
        symptoms_list = []
        st.error("Failed to fetch symptoms from Flask API.")
except Exception as e:
    symptoms_list = []
    st.error(f"Could not connect to Flask API: {e}")

# Select symptoms
selected_symptoms = st.multiselect(
    " Choose your symptoms:",
    options=symptoms_list,
    help="Start typing to search among all symptoms",
    placeholder="Type to search symptoms..."
)

st.caption(f"You selected **{len(selected_symptoms)}** symptoms out of {len(symptoms_list)} available.")

# Predict Button
if st.button(" Predict Disease"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        input_data = {sym: 1 if sym in selected_symptoms else 0 for sym in symptoms_list}
        try:
            with st.spinner("Analyzing your symptoms..."):
                resp = requests.post(f"{API_BASE}/predict", json=input_data)

            if resp.status_code == 200:
                result = resp.json().get("prediction", "Unknown")

                st.markdown(f"""
                <div style='background-color:#e8f5e9; border-left:6px solid #4CAF50; padding:15px; border-radius:8px;'>
                  <h3>Predicted Disease: <span style='color:#2E7D32;'>{result}</span></h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"API Error: {resp.text}")
        except Exception as e:
            st.error(f"Could not connect to Flask API: {e}")

# Health Tips Section
st.markdown("---")
st.subheader(" Health & Wellness Tips")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="background-color:#f8f9fa; padding:20px; border-radius:10px; box-shadow:2px 2px 8px #ddd;">
    <h4 style="color:#2E86C1;">🥦 Nutrition</h4>
    <p>Eat balanced meals with fruits, vegetables, proteins, and grains.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background-color:#f8f9fa; padding:20px; border-radius:10px; box-shadow:2px 2px 8px #ddd;">
    <h4 style="color:#27AE60;">🏋️ Exercise</h4>
    <p>Do at least 30 minutes of moderate physical activity daily.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background-color:#f8f9fa; padding:20px; border-radius:10px; box-shadow:2px 2px 8px #ddd;">
    <h4 style="color:#F39C12;">🧘 Mental Health</h4>
    <p>Practice meditation, journaling, or hobbies to maintain peace of mind.</p>
    </div>
    """, unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("""
    <div style="background-color:#f8f9fa; padding:20px; border-radius:10px; box-shadow:2px 2px 8px #ddd;">
    <h4 style="color:#8E44AD;">🛌 Rest</h4>
    <p>Get 7–8 hours of quality sleep for optimal recovery.</p>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div style="background-color:#f8f9fa; padding:20px; border-radius:10px; box-shadow:2px 2px 8px #ddd;">
    <h4 style="color:#C0392B;">🚭 Lifestyle</h4>
    <p>Avoid smoking, limit alcohol, and reduce stress.</p>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown("""
    <div style="background-color:#f8f9fa; padding:20px; border-radius:10px; box-shadow:2px 2px 8px #ddd;">
    <h4 style="color:#E67E22;">💧 Hydration</h4>
    <p>Drink 8–10 glasses of water each day to stay hydrated.</p>
    </div>
    """, unsafe_allow_html=True)