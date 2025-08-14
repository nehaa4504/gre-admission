import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

st.title("Graduate Admission Prediction")

# Path to model directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), "Model_Scaler")

# Check and load files
required_files = ['rf_model.pkl', 'xgb_model.pkl', 'scaler.pkl']
for file in required_files:
    if not os.path.exists(os.path.join(MODEL_DIR, file)):
        st.error(f"Required file {file} not found in Model_Scaler/")
        st.stop()

try:
    rf_model = pickle.load(open(os.path.join(MODEL_DIR, 'rf_model.pkl'), 'rb'))
    xgb_model = pickle.load(open(os.path.join(MODEL_DIR, 'xgb_model.pkl'), 'rb'))
    scaler = pickle.load(open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb'))
except Exception as e:
    st.error(f"Error loading model/scaler: {str(e)}")
    st.stop()

# Features
feature_names = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']

# Inputs
gre = st.number_input("GRE Score (260-340)", 260, 340, 300)
toefl = st.number_input("TOEFL Score (0-120)", 0, 120, 100)
univ_rating = st.number_input("University Rating (1-5)", 1, 5, 3)
sop = st.number_input("SOP Strength (1-5)", 1.0, 5.0, 3.0)
lor = st.number_input("LOR Strength (1-5)", 1.0, 5.0, 3.0)
cgpa = st.number_input("CGPA (6-10)", 6.0, 10.0, 8.0)
research = st.selectbox("Research Experience", [0, 1], index=0)

if st.button("Predict"):
    try:
        user_input = pd.DataFrame([[gre, toefl, univ_rating, sop, lor, cgpa, research]],
                                 columns=feature_names)
        user_scaled = scaler.transform(user_input)
        rf_pred = rf_model.predict(user_scaled)[0]
        xgb_pred = xgb_model.predict(user_scaled)[0]
        mean_pred = np.clip(np.mean([rf_pred, xgb_pred]), 0, 1)
        st.success(f"Predicted Chance of Admission: {round(mean_pred * 100, 2)}%")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
