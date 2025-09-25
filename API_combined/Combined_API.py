from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import re
import os
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import re

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
# Load BERT-based sentiment model
sentiment_model_path = os.path.join(BASE_DIR, "sentiment_model.pkl")
clf, tokenizer, le = joblib.load(sentiment_model_path)
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Function to get BERT embeddings
def get_bert_embedding(text):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

@app.route("/sentiment", methods=["POST"])
def predict_sentiment():
    try:
        data = request.get_json(force=True)

        # Accept either single text or multiple texts
        texts = data.get("texts") or [data.get("text")]
        if not texts or all(t.strip() == "" for t in texts):
            return jsonify({"error": "No text provided"}), 400

        results = []
        for text in texts:
            cleaned = re.sub(r"[^a-zA-Z\s]", "", text.lower())
            emb = get_bert_embedding(cleaned).reshape(1, -1)
            pred = clf.predict(emb)[0]
            sentiment = le.inverse_transform([pred])[0]
            results.append({"text": text, "sentiment": sentiment})

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------------------------------------
# 4) Customer Segmentation
# --------------------------------------------------
segmentation_model = joblib.load(
    os.path.join(BASE_DIR, "customer_segmentation_model.pkl")
)

cluster_labels = {
    0: "High Spenders",
    1: "Budget-Conscious",
    2: "Trend Seekers",
    3: "Loyal Mid-Lifers",
}
cluster_features = ["Purchases", "Spending", "Recency", "Response"]

# --------------------------
# Descriptive profiling logic
# --------------------------
def descriptive_profile(cluster_name, row):
    if cluster_name == "High Spenders":
        if row.get("Has_Children", 0) == 1:
            return "Wealthy, Family-Focused"
        else:
            return "Wealthy, Single / No Children"

    elif cluster_name == "Budget-Conscious":
        if row.get("Has_Children", 0) == 1:
            return "Budget-Savvy Family"
        else:
            return "Budget-Conscious Single"

    elif cluster_name == "Trend Seekers":
        if row.get("AgeGroup", 99) == 0:  # 0 = Young (18–25)
            return "Young Trend Seekers"
        else:
            return "Trend-Seeking Adult"

    elif cluster_name == "Loyal Mid-Lifers":
        if row.get("Has_Children", 0) == 1:
            return "Loyal, Family-Oriented"
        else:
            return "Loyal Adult / No Children"

    return cluster_name + " - Other"

@app.route("/predict-segment", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        # Extract only the trained features for clustering
        X = df[cluster_features]

        # Predict cluster (use segmentation_model, not pipeline!)
        cluster_id = int(segmentation_model.predict(X)[0])
        cluster_name = cluster_labels.get(cluster_id, "Unknown Cluster")

        # Use profiling rules
        profile = descriptive_profile(cluster_name, data)

        return jsonify({
            "Profile_Label": profile  # ✅ only send descriptive profile
        })

    except Exception as e:
        return jsonify({"error": str(e)})


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
