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
        edu_map = {
            "PhD": "Graduated",
            "Master": "Graduated",
            "Graduation": "Graduated",
            "2n Cycle": "2n Cycle",
            "Basic": "Basic"
        }
        if "Education" not in data.columns:
            data["Education"] = "Graduated"   # default
        data["Education"] = data["Education"].map(edu_map).fillna("Graduated")
        data["Education_Encoded"] = edu_encoder.transform(data["Education"])

        # --- Map Marital_Status ---
        marital_map = {
            "In Relationship": 1,
            "Single": 0
        }
        if "Marital_Status" not in data.columns:
            data["Marital_Status"] = "Single"   # default
        data["Marital_Status"] = data["Marital_Status"].map(marital_map).fillna(0)

        # --- Required features with defaults ---
        required_features = [
            "Age", "Education_Encoded", "Marital_Status",
            "Spending", "Purchases", "Complain", "Response",
            "Recency", "Income"
        ]

        defaults = {
            "Age": 35,
            "Spending": 0,
            "Purchases": 0,
            "Complain": 0,
            "Response": 0,
            "Recency": 50,
            "Income": 30000
        }

        for col in required_features:
            if col not in data.columns:
                data[col] = defaults.get(col, 0)
            data[col] = data[col].fillna(defaults.get(col, 0))

        # --- Predict ---
        preds = model.predict(data[required_features])
        data["Predicted_Segment"] = preds.tolist()

        return jsonify(data.to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
