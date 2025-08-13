import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import time

# --- App Title and Description ---
st.set_page_config(page_title="Startup Funding Predictor", page_icon="🚀")
st.title("🚀 Startup Funding Predictor")
st.markdown("""
This app demonstrates *Recursive Feature Elimination (RFE)* with a Random Forest Regressor to select the most important features for predicting startup funding amounts.

*How it works:*
1.  The indian_startup_founders.csv dataset is loaded.
2.  Categorical features are encoded into numbers.
3.  When you click the button below, the app will train a Random Forest model and use RFE to select the top 10 most influential features.
4.  Finally, it will display the selected features and the model's performance (Mean Squared Error).
""")

# --- Caching the data loading ---
@st.cache_data
def load_data():
    """Loads the dataset from the CSV file."""
    df = pd.read_csv('indian_startup_founders.csv')
    return df

df = load_data()

# --- The RFE and Model Training Logic ---
def train_model_and_get_features(data):
    """Performs RFE, trains the model, and returns results."""
    # Drop unnecessary columns
    df_processed = data.drop(['founder_id', 'startup_id'], axis=1)

    # Handle categorical variables
    categorical_cols = ['gender', 'education_level', 'alma_mater', 'field_of_study', 'is_first_generation', 'industry_sector', 'city', 'state', 'is_unicorn', 'has_female_investor', 'mentorship_access', 'policy_support']
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])

    # Define features (X) and target (y)
    X = df_processed.drop('funding_amount', axis=1)
    y = df_processed['funding_amount']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Initialize the base model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # 2. Initialize RFE to select the top 10 features
    rfe = RFE(estimator=model, n_features_to_select=10)

    # 3. Fit RFE
    rfe.fit(X_train, y_train)

    # 4. Get selected features and transform data
    selected_features = X_train.columns[rfe.support_]
    X_train_rfe = rfe.transform(X_train)
    X_test_rfe = rfe.transform(X_test)

    # 5. Train the final model on selected features
    model.fit(X_train_rfe, y_train)

    # 6. Make predictions and evaluate
    y_pred = model.predict(X_test_rfe)
    mse = mean_squared_error(y_test, y_pred)
    
    return selected_features, mse

# --- Interactive Button and Displaying Results ---
if st.button('Run Feature Elimination and Train Model'):
    with st.spinner('Performing RFE and training the model... This might take a moment.'):
        time.sleep(2) # To make the spinner more visible
        selected_features, mse = train_model_and_get_features(df)

    st.balloons()
    st.success("Analysis Complete!")
    
    st.subheader("📊 Most Important Features Selected by RFE")
    st.write("The following 10 features were found to be the most predictive of the startup's funding amount:")
    
    # Display features in a more readable format
    for i, feature in enumerate(selected_features):
        st.markdown(f"{i+1}.** {feature}")

    st.subheader("📈 Model Performance")
    st.metric(label="Mean Squared Error (MSE) on Test Set", value=f"{mse:,.2f}")
    st.info("The MSE represents the average squared difference between the estimated values and the actual values. A lower MSE indicates a better model fit.")

else:
    st.info("Click the button above to start the analysis.")
