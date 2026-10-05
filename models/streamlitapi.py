import pickle
import pandas as pd
import numpy as np
import streamlit as st


# ---------------------------------------
# Load model
# ---------------------------------------

model_package = pickle.load(
    open(
        "models/finlora_random_forest_deployment_package.pkl",
        "rb"
    )
)

model = model_package["model"]
preprocessor = model_package["preprocessor"]
model_features = model_package["model_features"]
threshold = model_package["threshold"]


# ---------------------------------------
# Streamlit application
# ---------------------------------------

def main():

    st.title("Finlora Fraud Detection and Risk Scoring")

    st.write(
        "Enter the transaction details to assess fraud risk."
    )

    st.divider()

    # ---------------------------------------
    # Transaction information
    # ---------------------------------------

    st.subheader("Transaction Details")

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=5000.0
    )

    avg_transaction_amount_30d = st.number_input(
        "Average Transaction Amount (30 Days)",
        min_value=0.0,
        value=5000.0
    )

    transaction_velocity_1h = st.number_input(
        "Transactions in Last 1 Hour",
        min_value=0,
        value=1
    )

    account_age_days = st.number_input(
        "Account Age (Days)",
        min_value=0,
        value=365
    )

    personal_spend_baseline_usd = st.number_input(
        "Personal Spending Baseline (USD)",
        min_value=0.0,
        value=5000.0
    )

    hour_of_day = st.number_input(
        "Hour of Transaction (0-23)",
        min_value=0,
        max_value=23,
        value=12
    )

    # ---------------------------------------
    # Categorical information
    # ---------------------------------------

    st.subheader("Transaction Information")

    account_type = st.selectbox(
        "Account Type",
        ["individual", "business"]
    )

    kyc_tier = st.selectbox(
        "KYC Tier",
        [
            "tier1_basic",
            "tier2_verified",
            "tier3_enhanced"
        ]
    )

    day_of_week = st.selectbox(
        "Day of Week",
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]
    )

    merchant_category = st.selectbox(
        "Merchant Category",
        [
            "atm withdrawal",
            "crypto exchange",
            "electronics",
            "gambling/gaming",
            "groceries",
            "healthcare",
            "insurance",
            "p2p transfer",
            "payroll transfer",
            "restaurants",
            "retail",
            "subscription/saas",
            "travel",
            "utilities",
            "wire transfer"
        ]
    )

    channel = st.selectbox(
        "Transaction Channel",
        [
            "api/integration",
            "card not present",
            "card present",
            "mobile app",
            "ussd",
            "web dashboard"
        ]
    )

    currency = st.selectbox(
        "Currency",
        ["NGN", "GBP", "USD", "EUR"]
    )

    transaction_country = st.selectbox(
        "Transaction Country",
        ["NG", "GB", "US", "DE", "FR"]
    )

    home_country = st.selectbox(
        "Home Country",
        ["NG", "GB", "US", "DE", "FR"]
    )

    account_age_group = st.selectbox(
        "Account Age Group",
        [
            "new",
            "recent",
            "mature",
            "established"
        ]
    )

    # ---------------------------------------
    # Risk indicators
    # ---------------------------------------

    st.subheader("Risk Indicators")

    is_cross_border = st.selectbox(
        "Cross-Border Transaction?",
        [0, 1]
    )

    is_new_device = st.selectbox(
        "New Device?",
        [0, 1]
    )

    negative_avg_flag = st.selectbox(
        "Negative Average Flag?",
        [0, 1]
    )

    device_status_unknown = st.selectbox(
        "Device Status Unknown?",
        [0, 1]
    )

    # ---------------------------------------
    # Prediction
    # ---------------------------------------

    if st.button("Assess Transaction Risk"):

        # Feature engineering

        if avg_transaction_amount_30d > 0:
            amount_to_avg_ratio = (
                amount /
                avg_transaction_amount_30d
            )
        else:
            amount_to_avg_ratio = 0

        log_amount = np.log1p(amount)

        log_amount_to_avg_ratio = np.log1p(
            amount_to_avg_ratio
        )

        day_numbers = {
            "Monday": 1,
            "Tuesday": 2,
            "Wednesday": 3,
            "Thursday": 4,
            "Friday": 5,
            "Saturday": 6,
            "Sunday": 7
        }

        day_of_week_num = day_numbers[day_of_week]

        day_sin = np.sin(
            2 * np.pi * day_of_week_num / 7
        )

        day_cos = np.cos(
            2 * np.pi * day_of_week_num / 7
        )

        hour_sin = np.sin(
            2 * np.pi * hour_of_day / 24
        )

        hour_cos = np.cos(
            2 * np.pi * hour_of_day / 24
        )

        high_amount_ratio_flag = (
            1 if amount_to_avg_ratio >= 3 else 0
        )

        high_velocity_flag = (
            1 if transaction_velocity_1h >= 5
            else 0
        )

        new_device_flag = is_new_device

        cross_border_flag = is_cross_border

        # ---------------------------------------
        # Create transaction
        # ---------------------------------------

        transaction = {

            "account_type": account_type,
            "kyc_tier": kyc_tier,
            "day_of_week": day_of_week,
            "hour_of_day": hour_of_day,
            "merchant_category": merchant_category,
            "channel": channel,

            "amount": amount,
            "currency": currency,

            "amount_to_avg_ratio":
                amount_to_avg_ratio,

            "avg_transaction_amount_30d":
                avg_transaction_amount_30d,

            "transaction_velocity_1h":
                transaction_velocity_1h,

            "transaction_country":
                transaction_country,

            "home_country":
                home_country,

            "is_cross_border":
                is_cross_border,

            "is_new_device":
                is_new_device,

            "account_age_days":
                account_age_days,

            "negative_avg_flag":
                negative_avg_flag,

            "personal_spend_baseline_usd":
                personal_spend_baseline_usd,

            "log_amount": log_amount,

            "log_amount_to_avg_ratio":
                log_amount_to_avg_ratio,

            "day_of_week_num":
                day_of_week_num,

            "day_sin": day_sin,

            "day_cos": day_cos,

            "hour_sin": hour_sin,

            "hour_cos": hour_cos,

            "high_amount_ratio_flag":
                high_amount_ratio_flag,

            "high_velocity_flag":
                high_velocity_flag,

            "new_device_flag":
                new_device_flag,

            "cross_border_flag":
                cross_border_flag,

            "account_age_group":
                account_age_group,

            "device_status_unknown":
                device_status_unknown
        }

        # ---------------------------------------
        # DataFrame
        # ---------------------------------------

        transaction_df = pd.DataFrame(
            [transaction],
            columns=model_features
        )

        # ---------------------------------------
        # Prediction
        # ---------------------------------------

        X = preprocessor.transform(
            transaction_df
        )

        fraud_probability = model.predict_proba(X)[0][1]

        fraud_percentage = (
            fraud_probability * 100
        )

        # ---------------------------------------
        # Results
        # ---------------------------------------

        st.divider()

        st.subheader("Risk Assessment")

        st.metric(
            "Fraud Probability",
            f"{fraud_percentage:.2f}%"
        )

        if fraud_probability >= threshold:

            st.error(
                "HIGH RISK — Potential Fraudulent Transaction"
            )

            st.write(
                "This transaction should be reviewed "
                "by a fraud analyst."
            )

        else:

            st.success(
                "LOW RISK — Transaction Appears Legitimate"
            )

            st.write(
                "The transaction is below the configured "
                "fraud-risk threshold."
            )


# ---------------------------------------
# Run application
# ---------------------------------------

if __name__ == "__main__":
    main()