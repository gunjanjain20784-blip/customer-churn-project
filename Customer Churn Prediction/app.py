import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")

st.title("Customer Churn Prediction")

st.write("Enter Customer Details:")

# Inputs
tenure = st.number_input("Tenure (months)", min_value=0, value=10)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=700.0)

st.write("Click predict to see result")

# Prediction
if st.button("Predict"):
    
    # 19 features (3 real + 16 dummy)
    input_data = [tenure, monthly_charges, total_charges] + [0]*16
    input_array = np.array([input_data])
    
    prediction = model.predict(input_array)
    probability = model.predict_proba(input_array)

    churn_prob = probability[0][1] * 100

    if prediction[0] == 1:
        st.error("Customer will churn ❌")
    else:
        st.success("Customer will not churn ✅")

    st.info(f"Churn Probability: {churn_prob:.2f}%")