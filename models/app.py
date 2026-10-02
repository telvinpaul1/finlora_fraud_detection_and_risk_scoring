# FINLORA FINANCE FLOW
# FRAUD DETECTION & RISK SCORING DASHBOARD
# Streamlit is used to create the web application.
import streamlit as st

# Pandas is used to work with transaction data.
import pandas as pd

# NumPy is used for numerical calculations.
import numpy as np

# Joblib loads the trained machine-learning models.
import joblib

# Used for displaying charts.
import matplotlib.pyplot as plt

#  PAGE CONFIGURATION
st.set_page_config(
    page_title="Finlora Fraud Risk Scoring",
    page_icon="💳",
    layout="wide"
)
#  APPLICATION TITLE
st.title("💳 Finlora Finance Flow")
st.subheader("Fraud Detection & Risk Scoring System")

st.write(
    """
    This prototype uses a machine-learning model to estimate the
    probability that a transaction is fraudulent.

    The Random Forest model is used as the primary risk-scoring model.
    Transactions can then be prioritised for further analyst review.
    """
)

#  LOAD TRAINED MODELS
# The models are stored in the 'models' folder.
MODEL_FOLDER = "models"
try:

    # Load the trained Random Forest model.
    rf_model = joblib.load(
        f"{MODEL_FOLDER}/random_forest_model.pkl"
    )

    # Load the Logistic Regression model.
    logistic_model = joblib.load(
        f"{MODEL_FOLDER}/logistic_regression_model.pkl"
    )

     # Load the preprocessing object used during training.
    preprocessor = joblib.load(
        f"{MODEL_FOLDER}/preprocessor.pkl"
    )

    # Load the scaler used by Logistic Regression.
    scaler = joblib.load(
        f"{MODEL_FOLDER}/scaler.pkl"
    )

    # Load the original model feature list.
    model_features = joblib.load(
        f"{MODEL_FOLDER}/model_features.pkl"
    )

    st.success("Models loaded successfully.")

except Exception as e:

    st.error(
        "Unable to load the trained models. "
        "Make sure the 'models' folder is in the same directory as app.py."
    )

    st.stop()

    #  SIDEBAR
    st.sidebar.header("Model Configuration")

# Allow the user to choose which trained model to use.
selected_model = st.sidebar.selectbox(
    "Select model",
    [
        "Random Forest",
        "Logistic Regression"
    ]
)

st.sidebar.write(
    """
    **Random Forest** is the primary model for this prototype.

    **Logistic Regression** is retained as the interpretable baseline.
    """
)

#  UPLOAD TRANSACTION DATA
st.header("Transaction Risk Assessment")

st.write(
    """
    Upload a CSV file containing transactions with the same feature
    structure used during model training.
    """
)

uploaded_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"]
)
#  PROCESS UPLOADED DATA