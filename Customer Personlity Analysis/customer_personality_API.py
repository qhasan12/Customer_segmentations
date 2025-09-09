from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Load saved ML components
scaler = joblib.load("Customer Personlity Analysis/segmentation_scaler.pkl")
model = joblib.load("Customer Personlity Analysis/segmentation_model.pkl")
segment_encoder = joblib.load("Customer Personlity Analysis/segment_encoder.pkl")

# Features required for prediction
required_features = [
    "Income", "TotalSpend", "TotalPurchases", "Effective_Campaigns",
    "Recency", "Age", "Family_Size", "Has_Children",
    "Is_In_Relationship", "Is_Single", "Education_Encoded"
]

app = Flask(__name__)

@app.route("/")
def home():
    return {"message": "Customer Segmentation API is running"}

@app.route("/customer-segment", methods=['POST'])
def customer_segment():
    try:
        # Parse input JSON
        data_json = request.get_json()
        data = pd.DataFrame(data_json)

        # Validate input
        missing_cols = [c for c in required_features if c not in data.columns]
        if missing_cols:
            return jsonify({"error": f"Missing columns: {missing_cols}"}), 400

        # Scale and predict
        X_scaled = scaler.transform(data[required_features])
        predictions = model.predict(X_scaled)
        segments = segment_encoder.inverse_transform(predictions)

        # Attach results
        data["Predicted_Segment"] = segments.tolist()

        return jsonify(data.to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
