import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Page Config ---
st.set_page_config(page_title="Insurance Premium Predictor", layout="centered")

# --- Load the Model ---
@st.cache_resource
def load_model():
    with open('modelliniear.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
st.title("🏥 Health Insurance Predictor")
st.write("Enter your details below to estimate your insurance charges.")
st.divider()

# --- Input Section ---
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=100, value=25)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    children = st.selectbox("Number of Children", options=[0, 1, 2, 3, 4, 5])

with col2:
    sex = st.selectbox("Sex", options=["Male", "Female"])
    smoker = st.selectbox("Smoker", options=["Yes", "No"])
    region = st.selectbox("Region", options=["Southwest", "Southeast", "Northwest", "Northeast"])

# --- Data Preprocessing ---
# Mapping inputs to match model expectations (Numerical encoding)
sex_val = 1 if sex == "Male" else 0
smoker_val = 1 if smoker == "Yes" else 0

# Simple mapping for region (Adjust this based on how you trained your model)
region_map = {"Southwest": 0, "Southeast": 1, "Northwest": 2, "Northeast": 3}
region_val = region_map[region]

# Create feature array
features = np.array([[age, sex_val, bmi, children, smoker_val, region_val]])

# --- Prediction Logic ---
st.divider()
if st.button("Calculate Predicted Charges"):
    prediction = model.predict(features)
    
    st.balloons()
    st.success(f"### Estimated Annual Premium: ${prediction[0]:,.2f}")
    
    # Extra Insight
    st.info("💡 **Note:** This is a linear regression estimate based on the historical data used to train the model.")

# --- Footer ---
st.caption("Developed by Shreyas Jagtap | Data Analysis Project 2026")
