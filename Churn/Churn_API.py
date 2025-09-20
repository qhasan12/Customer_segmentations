from flask import Flask, request, jsonify
import joblib
import numpy as np

# ------------------------
# Load trained model + scaler
# ------------------------
model, scaler = joblib.load("D:\work\Github\Customer_segmentations\Churn\churn_model.pkl")

app = Flask(__name__)

# Expected feature order (must match training)
FEATURES = [
    "recency",             # days since last order
    "frequency",           # number of orders
    "monetary",            # total spent
    "avg_payment_value",   # avg order value proxy
    "avg_review_score"
]

@app.route("/predict_churn", methods=["POST"])
def predict():
    try:
        # Get JSON data
        data = request.get_json()

        # Ensure all features exist
        input_data = []
        for feature in FEATURES:
            if feature not in data:
                return jsonify({"error": f"Missing feature: {feature}"}), 400
            input_data.append(data[feature])

        # Convert to numpy array
        input_array = np.array(input_data).reshape(1, -1)

        # Scale using the same scaler from training
        input_scaled = scaler.transform(input_array)

        # Predict churn (0/1) and probability
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0, 1]
        # prediction = int(probability >= 0.2)

        return jsonify({
            "churn_prediction": int(prediction),   # 0 = Not Churn, 1 = Churn
            "churn_probability": float(probability)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "Customer Churn Prediction API is running 🚀"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
