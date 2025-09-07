from flask import Flask, request, jsonify
import joblib
import pandas as pd
# import requests

app = Flask(__name__)

model = joblib.load("customer_segmentation_model.pkl")

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

        preds = model.predict(df)

        return jsonify({
            "input": df.to_dict(orient="records"),
            "predicted_segment": preds.tolist()
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    

    

if __name__ == "__main__":
    app.run(debug = True)

