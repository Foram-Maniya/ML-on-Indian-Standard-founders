import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder
import numpy as np

# --- App Configuration ---
st.set_page_config(page_title="Startup Funding Predictor", page_icon="🚀", layout="wide")


# --- Caching Functions for Efficiency ---

@st.cache_data
def load_data():
    """Loads and preprocesses the dataset, returning the raw and processed dataframes."""
    df = pd.read_csv('indian_startup_founders.csv')
    df_processed = df.drop(['founder_id', 'startup_id'], axis=1)
    return df, df_processed

@st.cache_resource
def get_encoders_and_model(df_processed):
    """
    Handles all data preparation, model training with RFE, and returns all necessary components.
    This function is cached to run only once.
    """
    # Create a dictionary to store label encoders for each categorical column
    encoders = {}
    categorical_cols = ['gender', 'education_level', 'alma_mater', 'field_of_study', 'is_first_generation', 'industry_sector', 'city', 'state', 'is_unicorn', 'has_female_investor', 'mentorship_access', 'policy_support']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        encoders[col] = le

    # Define features (X) and target (y)
    X = df_processed.drop('funding_amount', axis=1)
    y = df_processed['funding_amount']
    
    # --- RFE and Model Training ---
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    rfe = RFE(estimator=model, n_features_to_select=10)
    rfe.fit(X, y) # Fit RFE on the entire dataset to determine the best features
    
    # Get the final set of selected features and train the definitive model
    selected_features = X.columns[rfe.support_]
    X_selected = X[selected_features]
    
    final_model = RandomForestRegressor(n_estimators=100, random_state=42)
    final_model.fit(X_selected, y)
    
    return final_model, selected_features, encoders, X_selected.columns

# --- Main Application ---

# Load data and train model
df_raw, df_processed = load_data()
model, selected_features, encoders, trained_columns = get_encoders_and_model(df_processed.copy()) # Use a copy to be safe

# --- UI Layout ---

# Header
st.title("🚀 Startup Funding Predictor")
st.markdown("Enter your startup's details in the sidebar to predict its potential funding amount. The prediction is based on a Random Forest model trained on a dataset of Indian startup founders.")

# --- Sidebar for User Input ---
st.sidebar.header("Enter Your Startup's Details")

input_data = {}
# Create input fields dynamically based on selected features
for feature in trained_columns:
    if feature in encoders: # Categorical feature
        # Use the raw dataframe to get original string values for the dropdown
        unique_values = df_raw[feature].unique()
        input_data[feature] = st.sidebar.selectbox(f"Select {feature.replace('_', ' ').title()}", unique_values)
    else: # Numerical feature
        # Provide sensible min/max values for number inputs
        min_val = float(df_raw[feature].min())
        max_val = float(df_raw[feature].max())
        mean_val = float(df_raw[feature].mean())
        input_data[feature] = st.sidebar.number_input(f"Enter {feature.replace('_', ' ').title()}", min_value=min_val, max_value=max_val, value=mean_val)

# --- Prediction Logic ---
if st.sidebar.button("Predict Funding Amount", type="primary"):
    # Create a dataframe from user inputs
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical features using the stored encoders
    for feature in encoders:
        if feature in input_df.columns:
            le = encoders[feature]
            # Handle cases where a value might not have been seen during fit
            input_df[feature] = input_df[feature].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1) # Use -1 for unseen labels

    # Ensure the columns are in the same order as during training
    input_df = input_df[trained_columns]
    
    # Make the prediction
    prediction = model.predict(input_df)[0]
    
    st.subheader("🎉 Predicted Funding Amount")
    st.metric(label="Estimated Funding (in USD)", value=f"${prediction:,.2f}")
    st.info("Note: This prediction is based on a machine learning model and should be considered an estimate, not a guarantee.")

# --- Feature Analysis Section (Collapsible) ---
with st.expander("🔬 Click here to see the Model's Feature Analysis"):
    st.subheader("📊 Most Important Features for Prediction")
    st.write("The model identified these 10 features as the most important for predicting funding amounts:")
    
    # Display features in a more readable format
    for i, feature in enumerate(selected_features):
        st.markdown(f"{i+1}.** {feature}")
        
    st.warning("This analysis shows which factors the model weighs most heavily. It is not financial advice.")
