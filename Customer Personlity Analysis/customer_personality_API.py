from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Load saved models
scaler = joblib.load("marketing_scaler.pkl")
kmeans = joblib.load("marketing_kmeans_model.pkl")
cluster_labels = joblib.load("marketing_cluster_lables.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return {"message": "API is running"}

@app.route("/marketing", methods=['POST'])
def marketing():
    try:
        data_json = request.get_json()
        data = pd.DataFrame(data_json)

        # Check required columns
        required_cols = ["Income", "TotalSpend"]
        missing_cols = [c for c in required_cols if c not in data.columns]
        if missing_cols:
            return jsonify({"error": f"Missing columns: {missing_cols}"}), 400

        # Preprocess numeric features
        X = data[required_cols].copy()
        X_scaled = scaler.transform(X)

        # Predict clusters
        clusters = kmeans.predict(X_scaled)
        segments = [cluster_labels[c] for c in clusters]

        # Add predictions
        data["Cluster"] = clusters
        data["Segment"] = segments

        return jsonify(data.to_dict(orient='records'))
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
