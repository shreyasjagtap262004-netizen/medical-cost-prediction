import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Page Config ---
st.set_page_config(page_title="Insurance Premium Predictor", layout="centered")

# --- Load the Model ---
@st.cache_resource
def load_model():
    # Ensure the filename matches your uploaded file exactly
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
        background-color: #28a745;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
st.title("🏥 Medical Insurance Cost Predictor")
st.write("Professional Data Analysis Tool for Premium Estimation")
st.divider()

# --- Input Section ---
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=100, value=25)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    children = st.selectbox("Number of Children", options=[0, 1, 2, 3, 4, 5])

with col2:
    sex = st.selectbox("Sex", options=["Male", "Female"])
    smoker = st.selectbox("Smoker", options=["Yes", "No"])
    region = st.selectbox("Region", options=["Southwest", "Southeast", "Northwest", "Northeast"])

# --- Data Preprocessing ---
# Standard numerical encoding used in data analyst workflows
sex_val = 1 if sex == "Male" else 0
smoker_val = 1 if smoker == "Yes" else 0

# Mapping region to numerical values (0-3)
region_map = {"Southwest": 0, "Southeast": 1, "Northwest": 2, "Northeast": 3}
region_val = region_map[region]

# Create feature array in the exact order: age, sex, bmi, children, smoker, region
features = np.array([[age, sex_val, bmi, children, smoker_val, region_val]])

# --- Prediction Logic ---
st.divider()
if st.button("Calculate Predicted Charges"):
    try:
        prediction = model.predict(features)
        
        # --- THE FIX ---
        # Using .item() extracts the specific number from the NumPy array 
        # so that it can be formatted with commas and decimals.
        final_result = prediction.item()
        
        st.balloons()
        st.success(f"### Estimated Annual Premium: ${final_result:,.2f}")
        
        # Displaying the breakdown for the user
        with st.expander("See Input Details"):
            st.write(f"**Age:** {age} | **BMI:** {bmi} | **Smoker:** {smoker}")
            st.write(f"**Processed Features for Model:** `{features.tolist()}`")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# --- Footer ---
st.caption("Developed by Shreyas Jagtap | Aspiring Data Analyst 2026")
