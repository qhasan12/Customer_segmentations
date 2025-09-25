# from flask import Flask, request, jsonify
# import pandas as pd
# import joblib

# # Load trained pipeline (preprocessing + classifier)
# model = joblib.load(r"D:\work\Github\Customer_segmentations\Simple Segmentation\customer_segmentation_pipeline.pkl")

# # Cluster labels mapping
# cluster_labels = {
#     0: "High Spenders",
#     1: "Budget-Conscious",
#     2: "Trend Seekers",
#     3: "Loyal Mid-Lifers"
# }

# # Features expected by the pipeline
# required_features = [
#     "AgeGroup", "Education_Encoded", "Marital_Status",
#     "Income", "Has_Children",
#     "Purchases", "Spending",
#     "Recency", "Response"
# ]

# # Default values (in case fields are missing in input)
# defaults = {
#     "AgeGroup": 1,
#     "Education_Encoded": 0,
#     "Marital_Status": 0,
#     "Income": 0,
#     "Has_Children": 0,
#     "Purchases": 0,
#     "Spending": 0,
#     "Recency": 0,
#     "Response": 0
# }

# app = Flask(__name__)

# @app.route("/")
# def home():
#     return {"message": "Customer Segmentation API is running"}

# @app.route("/predict-segment", methods=["POST"])
# def predict_segment():
#     try:
#         # Parse JSON input
#         data_json = request.get_json()
#         data = pd.DataFrame(data_json)

#         # Ensure all required features are present
#         for col in required_features:
#             if col not in data.columns:
#                 data[col] = defaults[col]
#             data[col] = data[col].fillna(defaults[col])

#         # Predict clusters
#         preds = model.predict(data[required_features])
#         data["Predicted_Cluster"] = preds
#         data["Cluster_Label"] = data["Predicted_Cluster"].map(cluster_labels)

#         return jsonify(data.to_dict(orient="records"))

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# if __name__ == "__main__":
#     app.run(debug=True)




from flask import Flask, request, jsonify
import pandas as pd
import joblib

# --------------------------
# Load trained model + pipeline
# --------------------------
pipeline = joblib.load(r"D:\work\Github\Customer_segmentations\Simple Segmentation\customer_segmentation_model.pkl")

# Cluster label mapping
cluster_labels = {
    0: "High Spenders",
    1: "Budget-Conscious",
    2: "Trend Seekers",
    3: "Loyal Mid-Lifers"
}

# Features the model was trained on
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

# --------------------------
# Flask app
# --------------------------
app = Flask(__name__)

@app.route("/predict-segment", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        # Extract only the trained features for clustering
        X = df[cluster_features]

        # Predict cluster
        cluster_id = int(pipeline.predict(X)[0])
        cluster_name = cluster_labels.get(cluster_id, "Unknown Cluster")

        # Use profiling rules
        profile = descriptive_profile(cluster_name, data)

        return jsonify({
            "Profile_Label": profile  # ✅ only send descriptive profile
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
