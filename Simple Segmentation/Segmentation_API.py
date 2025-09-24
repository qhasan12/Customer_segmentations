from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Load trained pipeline (preprocessing + classifier)
model = joblib.load(r"D:\work\Github\Customer_segmentations\Simple Segmentation\customer_segmentation_pipeline.pkl")

# Cluster labels mapping
cluster_labels = {
    0: "High Spenders",
    1: "Budget-Conscious",
    2: "Trend Seekers",
    3: "Loyal Mid-Lifers"
}

# Features expected by the pipeline
required_features = [
    "AgeGroup", "Education_Encoded", "Marital_Status",
    "Income", "Has_Children",
    "Purchases", "Spending",
    "Recency", "Response"
]

# Default values (in case fields are missing in input)
defaults = {
    "AgeGroup": 1,
    "Education_Encoded": 0,
    "Marital_Status": 0,
    "Income": 0,
    "Has_Children": 0,
    "Purchases": 0,
    "Spending": 0,
    "Recency": 0,
    "Response": 0
}

app = Flask(__name__)

@app.route("/")
def home():
    return {"message": "Customer Segmentation API is running"}

@app.route("/predict-segment", methods=["POST"])
def predict_segment():
    try:
        # Parse JSON input
        data_json = request.get_json()
        data = pd.DataFrame(data_json)

        # Ensure all required features are present
        for col in required_features:
            if col not in data.columns:
                data[col] = defaults[col]
            data[col] = data[col].fillna(defaults[col])

        # Predict clusters
        preds = model.predict(data[required_features])
        data["Predicted_Cluster"] = preds
        data["Cluster_Label"] = data["Predicted_Cluster"].map(cluster_labels)

        return jsonify(data.to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
