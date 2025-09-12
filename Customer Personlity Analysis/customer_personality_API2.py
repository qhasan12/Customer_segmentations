from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Load ML model and encoder
model = joblib.load(r"D:\work\Github\Customer_segmentations\Customer Personlity Analysis\segment_classifier.pkl")
edu_encoder = joblib.load(r"D:\work\Github\Customer_segmentations\Customer Personlity Analysis\edu_encoder.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return {"message": "Customer Segmentation API is running"}

@app.route("/predict-segment", methods=["POST"])
def predict_segment():
    try:
        data_json = request.get_json()
        data = pd.DataFrame(data_json)

        # --- Normalize Education values ---
        if "Education" in data.columns:
            edu_map = {
                "PhD": "Graduated",
                "Master": "Graduated",
                "Graduation": "Graduated",
                "2n Cycle": "2n Cycle",
                "Basic": "Basic"
            }
            data["Education"] = data["Education"].map(edu_map).fillna(data["Education"])
            data["Education_Encoded"] = edu_encoder.transform(data["Education"])
        else:
            return jsonify({"error": "Missing column: Education"}), 400

        # --- Map Marital_Status ---
        if "Marital_Status" in data.columns:
            marital_map = {
                "In Relationship": 1,
                "Single": 0
            }
            data["Marital_Status"] = data["Marital_Status"].map(marital_map)
        else:
            return jsonify({"error": "Missing column: Marital_Status"}), 400

        # --- Ensure all required features are there ---
        required_features = [
            "Age", "Education_Encoded", "Marital_Status",
            "Spending", "Purchases", "Complain", "Response",
            "Recency", "Income"
        ]

        missing_cols = [c for c in required_features if c not in data.columns]
        if missing_cols:
            return jsonify({"error": f"Missing columns: {missing_cols}"}), 400

        # --- Predict ---
        preds = model.predict(data[required_features])
        data["Predicted_Segment"] = preds.tolist()

        return jsonify(data.to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
