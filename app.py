import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Startup Job Creation Predictor",
    page_icon="🚀",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Asset Loading ---
# Using st.cache_data to prevent reloading assets on every interaction.
@st.cache_data
def load_asset(path):
    """Loads a joblib file from the specified path."""
    try:
        with open(path, 'rb') as file:
            asset = joblib.load(file)
        return asset
    except FileNotFoundError:
        st.error(f"Asset file not found at '{path}'. Please ensure the file is in the same directory as app.py.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading asset from '{path}': {e}")
        return None

# --- Main Application ---
def run():
    """Main function to run the Streamlit application."""

    # --- Header ---
    st.title("Startup Job Creation Predictor 🚀")
    st.markdown("Enter the startup's details below to predict the number of jobs it is likely to create. This tool uses a machine learning model trained on data from Indian startups.")

    # --- Load Model and Scaler ---
    # Make sure 'model.pkl' and 'scaler.pkl' are in the same folder
    model = load_asset('model.pkl')
    scaler = load_asset('scaler.pkl')

    # Proceed only if both the model and scaler were loaded successfully
    if model and scaler:
        # --- User Input Section ---
        st.sidebar.header("Input Features")
        st.sidebar.markdown("Adjust the sliders and inputs to match the startup's profile.")

        age = st.sidebar.slider(
            "Founder's Age",
            min_value=18, max_value=80, value=30,
            help="The age of the primary founder at the time of founding."
        )

        prior_work_experience = st.sidebar.slider(
            "Prior Work Experience (in years)",
            min_value=0, max_value=50, value=5,
            help="Total years of professional experience the founder had before starting the company."
        )

        funding_amount = st.sidebar.number_input(
            "Total Funding Amount (in INR)",
            min_value=0.0, value=5000000.0, step=100000.0, format="%.2f",
            help="The total amount of funding received by the startup in Indian Rupees."
        )

        investor_count = st.sidebar.slider(
            "Number of Investors",
            min_value=0, max_value=50, value=4,
            help="The total number of investors who have funded the startup."
        )

        founding_year = st.sidebar.slider(
            "Founding Year",
            min_value=1990, max_value=2024, value=2018,
            help="The year the startup was founded."
        )

        # --- Prediction Logic ---
        if st.button("Predict Jobs Created", type="primary"):
            # The keys MUST exactly match the feature names the model was trained on, in the correct order.
            features = {
                'age': age,
                'prior_work_experience': prior_work_experience,
                'funding_amount': funding_amount,
                'investor_count': investor_count,
                'founding_year': founding_year
            }
            features_df = pd.DataFrame([features])

            # --- THE CRITICAL STEP ---
            # Scale the user's input features using the loaded scaler
            try:
                scaled_features = scaler.transform(features_df)
                
                # Make a prediction using the model on the SCALED features
                prediction = model.predict(scaled_features)
                
                # The model might return a float, so we round it to the nearest whole number
                predicted_jobs = int(np.round(prediction[0]))

                # --- Display Result ---
                st.success(f"*Predicted Number of Jobs Created: {predicted_jobs}*")
                st.balloons()

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

# --- Entry Point ---
if __name__ == "__main__":
    run()
