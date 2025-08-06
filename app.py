import streamlit as st
import pickle
import pandas as pd
import joblib

with open('model.pkl', 'rb') as f:
    model = joblib.load(f)

# Define the features used in the model
selected_features = ['industry_sector_Software', 'state_Other', 'mentorship_access_Yes', 'policy_support_No', 'age', 'prior_work_experience', 'funding_amount', 'funding_rounds', 'investor_count', 'founding_year']

# Streamlit App UI
st.title("📈 Prediction of jobs created ")

st.markdown(
    """
    This app predicts the *Jobs Created* based on Indian Standard Founders
    """
)

# Collect input data from user
input_data = {}
for feature in selected_features:
    input_data[feature] = st.number_input(f"Enter {feature}", value=0.0)

# Convert input into DataFrame
input_df = pd.DataFrame([input_data])

# Predict when button is pressed
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.success(f"📊 Predicted Jobs Created : *{prediction[0]:.2f}%*")