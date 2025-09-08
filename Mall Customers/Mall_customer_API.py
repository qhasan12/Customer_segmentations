from flask import Flask, request, jsonify
import joblib
import pandas as pd
# import requests

app = Flask(__name__)

scaler = joblib.load("scaler.pkl")
kmeans = joblib.load("kmeans_model.pkl")
cluster_labels = joblib.load("cluster_lables.pkl")

@app.route("/")
def home():
    return {"message": "API is running"}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        data = request.get_json(force=True)
        print("Raw request data:", request.data)
        print("Parsed JSON:", data)

        df = pd.DataFrame(data if isinstance(data, list) else [data])
        
        required_cols = ['Annual Income (k$)', 'Spending Score (1-100)']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return jsonify({"error" : f"Missing Columns: {missing}"}), 400
        
        scaled = scaler.transform(df[required_cols])
        cluster_pred = kmeans.predict(scaled)

        segments = [cluster_labels[c] for c in cluster_pred]

        return jsonify({
            "input": df.to_dict(orient = "records"),
            "predicted_cluster": cluster_pred.tolist(),
            "predicted_segment": segments
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug = True)

