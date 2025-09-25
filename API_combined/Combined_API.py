from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import re
import os

# --------------------------------------------------
# Initialize app
# --------------------------------------------------
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def validate_fields(data, required_fields):
    """Check for missing required fields in request JSON."""
    missing = [f for f in required_fields if f not in data]
    if missing:
        return False, {"error": f"Missing fields: {missing}"}
    return True, None


# --------------------------------------------------
# 1) Churn Prediction
# --------------------------------------------------
churn_model, churn_scaler = joblib.load(
    os.path.join(BASE_DIR, "churn_model.pkl")
)

CHURN_FEATURES = ["recency", "frequency", "monetary", "avg_payment_value", "avg_review_score"]

@app.route("/predict-churn", methods=["POST"])
def predict_churn():
    try:
        data = request.get_json()
        valid, error = validate_fields(data, CHURN_FEATURES)
        if not valid:
            return jsonify(error), 400

        input_array = np.array([data[f] for f in CHURN_FEATURES]).reshape(1, -1)
        input_scaled = churn_scaler.transform(input_array)

        prediction = churn_model.predict(input_scaled)[0]
        probability = churn_model.predict_proba(input_scaled)[0, 1]

        return jsonify({
            "churn_prediction": int(prediction),
            "churn_probability": float(probability),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------------------------------------
# 2) Product Mining Recommendation
# --------------------------------------------------
rules_product = joblib.load(os.path.join(BASE_DIR, "rules_product.pkl"))
rules_aisle = joblib.load(os.path.join(BASE_DIR, "rules_aisle.pkl"))
rules_department = joblib.load(os.path.join(BASE_DIR, "rules_department.pkl"))

def recommend(cart_items, rules, top_n=5):
    cart_items = set(cart_items)
    recs = []
    for _, row in rules.iterrows():
        if row["antecedents"].issubset(cart_items):
            for consequent in row["consequents"]:
                if consequent not in cart_items:
                    reason = (f"Because you bought {', '.join(row['antecedents'])}, "
                              f"customers also often buy {consequent}")
                    recs.append({
                        "item": consequent,
                        "reason": reason,
                        "confidence": float(row["confidence"]),
                        "lift": float(row["lift"]),
                    })
    recs = sorted(recs, key=lambda x: (x["confidence"], x["lift"]), reverse=True)

    seen, final = set(), []
    for r in recs:
        if r["item"] not in seen:
            final.append(r)
            seen.add(r["item"])
        if len(final) >= top_n:
            break
    return final

@app.route("/recommend", methods=["POST"])
def recommend_all():
    try:
        data = request.json
        cart_items = data.get("cart", [])
        result = {
            "product_recommendations": recommend(cart_items, rules_product),
            "aisle_recommendations": recommend(cart_items, rules_aisle),
            "department_recommendations": recommend(cart_items, rules_department),
        }

        # Console log the response
        print("Request cart_items:", cart_items)
        print("Response:", result)

        return jsonify(result)
    except Exception as e:
        print("Error in /recommend:", e)
        return jsonify({"error": str(e)}), 500



# --------------------------------------------------
# 3) Sentiment Analysis
# --------------------------------------------------
sentiment_model, sentiment_vectorizer = joblib.load(
    os.path.join(BASE_DIR, "sentiment_model.pkl")
)

@app.route("/sentiment", methods=["POST"])
def predict_sentiment():
    try:
        data = request.get_json(force=True)
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "No text provided"}), 400

        cleaned = re.sub(r"[^a-zA-Z\s]", "", text.lower())
        X_input = sentiment_vectorizer.transform([cleaned])
        sentiment = sentiment_model.predict(X_input)[0]

        return jsonify({"sentiment": sentiment})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------------------------------------
# 4) Customer Segmentation
# --------------------------------------------------
segmentation_model = joblib.load(
    os.path.join(BASE_DIR, "customer_segmentation_pipeline.pkl")
)

cluster_labels = {
    0: "High Spenders",
    1: "Budget-Conscious",
    2: "Trend Seekers",
    3: "Loyal Mid-Lifers",
}

SEGMENT_FEATURES = [
    "AgeGroup", "Education_Encoded", "Marital_Status",
    "Income", "Has_Children",
    "Purchases", "Spending",
    "Recency", "Response",
]

SEGMENT_DEFAULTS = {
    "AgeGroup": 1, "Education_Encoded": 0, "Marital_Status": 0,
    "Income": 0, "Has_Children": 0,
    "Purchases": 0, "Spending": 0,
    "Recency": 0, "Response": 0,
}

@app.route("/predict-segment", methods=["POST"])
def predict_segment():
    try:
        data_json = request.get_json()
        data = pd.DataFrame(data_json)

        for col in SEGMENT_FEATURES:
            if col not in data.columns:
                data[col] = SEGMENT_DEFAULTS[col]
            data[col] = data[col].fillna(SEGMENT_DEFAULTS[col])

        preds = segmentation_model.predict(data[SEGMENT_FEATURES])
        data["Predicted_Cluster"] = preds
        data["Cluster_Label"] = data["Predicted_Cluster"].map(cluster_labels)

        return jsonify(data.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------------------------------------
# Root
# --------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return {
        "message": "Unified Customer Analytics API is running 🚀",
        "endpoints": [
            "/predict-churn",
            "/recommend",
            "/sentiment",
            "/predict-segment",
        ],
    }


# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
