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

# --- Model Loading ---
# This function loads the trained model.
# It's wrapped in st.cache_data to prevent reloading on every interaction.
@st.cache_data
def load_model(path):
    """Loads a machine learning model from a .pkl file."""
    try:
        with open(path, 'rb') as file:
            model = joblib.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at '{path}'. Please ensure the model file is in the same directory as app.py.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

# --- Main Application ---
def run():
    """Main function to run the Streamlit application."""

    # --- Header ---
    st.title("Startup Job Creation Predictor 🚀")
    st.markdown("Enter the startup's details below to predict the number of jobs it is likely to create. This tool uses a machine learning model trained on data from Indian startups.")

    # --- Load Model ---
    # Make sure your trained model is saved as 'model.pkl' in the same folder
    model = load_model('model.pkl')

    # Proceed only if the model was loaded successfully
    if model:
        # --- User Input Section ---
        st.sidebar.header("Input Features")
        st.sidebar.markdown("Adjust the sliders and inputs to match the startup's profile.")

        # Create input fields in the sidebar for the features your model needs
        # The default values are set to be reasonable starting points.
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
        # Create a button in the main area to trigger the prediction
        if st.button("Predict Jobs Created", type="primary"):
            # Create a dictionary of the user's inputs
            # The keys MUST exactly match the feature names your model was trained on
            features = {
                'age': age,
                'prior_work_experience': prior_work_experience,
                'funding_amount': funding_amount,
                'investor_count': investor_count,
                'founding_year': founding_year
            }

            # Convert the dictionary to a pandas DataFrame
            features_df = pd.DataFrame([features])

            # Make a prediction using the loaded model
            try:
                prediction = model.predict(features_df)
                
                # The model might return a float, so we round it to the nearest whole number
                # as you can't have a fraction of a job.
                predicted_jobs = int(np.round(prediction[0]))

                # --- Display Result ---
                st.success(f"*Predicted Number of Jobs Created: {predicted_jobs}*")
                st.balloons()

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

# --- Entry Point ---
# This ensures the run() function is called when the script is executed
if __name__ == "__main__":

    run()
