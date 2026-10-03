
# Transaction scoring
# FILE: src/scoring.py

def score_transaction(transaction, model_package):

    model = model_package["model"]
    preprocessor = model_package["preprocessor"]
    model_features = model_package["model_features"]
    threshold = model_package["threshold"]

    transaction_df = pd.DataFrame([transaction])

    missing_features = [
        col for col in model_features
        if col not in transaction_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    transaction_df = transaction_df[model_features]

    transaction_encoded = preprocessor.transform(
        transaction_df
    )

    fraud_probability = model.predict_proba(
        transaction_encoded
    )[0, 1]

    risk_score = fraud_probability * 100

    review_required = fraud_probability >= threshold

    if fraud_probability >= 0.50:
        risk_category = "High Risk"
    elif fraud_probability >= 0.30:
        risk_category = "Medium Risk"
    else:
        risk_category = "Low Risk"

    return {
        "fraud_probability": round(fraud_probability, 4),
        "risk_score": round(risk_score, 2),
        "risk_category": risk_category,
        "review_required": review_required,
        "operating_threshold": threshold
    }

# Review queue
def create_review_queue(transactions, model_package):

    model = model_package["model"]
    preprocessor = model_package["preprocessor"]
    model_features = model_package["model_features"]
    threshold = model_package["threshold"]

    X_transactions = transactions[model_features]

    X_encoded = preprocessor.transform(
        X_transactions
    )

    probabilities = model.predict_proba(
        X_encoded
    )[:, 1]

    results = transactions[
        ["transaction_id"]
    ].copy()

    results["fraud_probability"] = probabilities

    results["risk_score"] = probabilities * 100

    results["risk_category"] = pd.cut(
        probabilities,
        bins=[
            -float("inf"),
            0.30,
            0.50,
            float("inf")
        ],
        labels=[
            "Low Risk",
            "Medium Risk",
            "High Risk"
        ]
    )

    results["review_required"] = (
        probabilities >= threshold
    )

    review_queue = results[
        results["review_required"] == True
    ].sort_values(
        "fraud_probability",
        ascending=False
    )

    return review_queue