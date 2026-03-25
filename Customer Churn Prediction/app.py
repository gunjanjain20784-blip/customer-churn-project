import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")

st.title("Customer Churn Prediction")

st.write("Demo Prediction")

if st.button("Predict"):
    
    # 19 dummy inputs (same as model features)
    input_data = np.zeros((1, 19))
    
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Customer will churn ❌")
    else:
        st.success("Customer will not churn ✅")