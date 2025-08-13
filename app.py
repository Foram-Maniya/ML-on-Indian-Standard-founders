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
@st.cache_data
def load_asset(path):
    """Loads a joblib file from the specified path."""
    try:
        with open(path, 'rb') as file:
            asset = joblib.load(file)
        return asset
    except FileNotFoundError:
        st.error(f"Asset file not found at '{path}'. Please ensure the file is in your project directory.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading asset from '{path}': {e}")
        return None

# --- Main Application ---
def run():
    """Main function to run the Streamlit application."""
    st.title("Startup Job Creation Predictor 🚀")
    st.markdown("Enter the startup's details below to predict the number of jobs it is likely to create.")

    # --- Load All Assets ---
    model = load_asset('model.pkl')
    scaler = load_asset('scaler.pkl')
    model_columns = load_asset('model_columns.pkl')

    # Proceed only if all assets were loaded successfully
    if model and scaler and model_columns:
        st.sidebar.header("Input Features")
        st.sidebar.markdown("Adjust the inputs to match the startup's profile.")

        # --- Numerical Inputs ---
        age = st.sidebar.slider("Founder's Age", 18, 80, 30)
        prior_work_experience = st.sidebar.slider("Prior Work Experience (years)", 0, 50, 5)
        funding_amount = st.sidebar.number_input("Total Funding Amount (INR)", min_value=0.0, value=5000000.0, step=100000.0)
        investor_count = st.sidebar.slider("Number of Investors", 0, 50, 4)
        founding_year = st.sidebar.slider("Founding Year", 1990, 2024, 2018)

        # --- Categorical Inputs ---
        # Unique values taken from your provided CSV
        city_options = ['Bengaluru', 'Other', 'Mumbai', 'Delhi', 'Gurugram']
        alma_mater_options = ['None', 'Other', 'IIT Delhi', 'BITS Pilani', 'IIT Roorkee', 'IIT Bombay', 'IIT Kharagpur', 'IIT Madras', 'IIM Ahmedabad']
        
        city = st.sidebar.selectbox("City", options=city_options)
        alma_mater = st.sidebar.selectbox("Founder's Alma Mater", options=alma_mater_options)

        # --- Prediction Logic ---
        if st.button("Predict Jobs Created", type="primary"):
            # 1. Create a dictionary of all inputs
            raw_features = {
                'age': age,
                'prior_work_experience': prior_work_experience,
                'funding_amount': funding_amount,
                'investor_count': investor_count,
                'founding_year': founding_year,
                'city': city,
                'alma_mater': alma_mater
            }
            input_df = pd.DataFrame([raw_features])

            # 2. One-Hot Encode the categorical features
            # This creates columns like 'city_Bengaluru', 'alma_mater_IIT Delhi', etc.
            encoded_input_df = pd.get_dummies(input_df, columns=['city', 'alma_mater'])

            # 3. Align columns with the model's training columns
            # This is the crucial step to fix the error. It ensures the DataFrame has the exact
            # same columns as the one the model was trained on.
            final_input_df = encoded_input_df.reindex(columns=model_columns, fill_value=0)

            # 4. Scale the final DataFrame
            try:
                scaled_features = scaler.transform(final_input_df)
                
                # 5. Make a prediction
                prediction = model.predict(scaled_features)
                predicted_jobs = int(np.round(prediction[0]))

                # --- Display Result ---
                st.success(f"*Predicted Number of Jobs Created: {predicted_jobs}*")
                st.balloons()

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

# --- Entry Point ---
if __name__ == "__main__":
    run()
