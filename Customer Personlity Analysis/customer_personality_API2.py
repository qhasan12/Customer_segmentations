from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Load trained model and encoder
model = joblib.load(r"D:\work\Github\Customer_segmentations\Customer Personlity Analysis\segment_classifier.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return {"message": "Customer Segmentation API is running"}

@app.route("/predict-segment", methods=["POST"])
def predict_segment():
    try:
        # Parse JSON
        data_json = request.get_json()
        data = pd.DataFrame(data_json)

        # --- Required features (model was trained on all of these) ---
        required_features = [
            "Age", "Education_Encoded", "Marital_Status",
            "Spending", "Purchases", "Complain", "Response",
            "Recency", "Income"
        ]

        # Defaults (demographics auto-filled here)
        defaults = {
            "Age": 35,
            "Education_Encoded": 0,   # assume Graduated
            "Marital_Status": 0,      # assume Single
            "Spending": 0,
            "Purchases": 0,
            "Complain": 0,
            "Response": 0,
            "Recency": 50,
            "Income": 30000
        }

        # Fill missing columns
        for col in required_features:
            if col not in data.columns:
                data[col] = defaults[col]
            data[col] = data[col].fillna(defaults[col])

        # --- Predict ---
        preds = model.predict(data[required_features])
        data["Predicted_Segment"] = preds.tolist()

        return jsonify(data.to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
