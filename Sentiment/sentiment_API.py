from flask import Flask, request, jsonify
import joblib
import re
import os
from flask_cors import CORS

# -------------------------
# Load model + vectorizer
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the path to the model file
MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.pkl")

# Load model + vectorizer
model, vectorizer = joblib.load(MODEL_PATH)
# -------------------------
# Initialize Flask
# -------------------------
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
# -------------------------
# Predict endpoint
# -------------------------
@app.route("/sentiment", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Clean text
    cleaned = re.sub(r'[^a-zA-Z\s]', '', text.lower())

    # Vectorize
    X_input = vectorizer.transform([cleaned])

    # Predict
    sentiment = model.predict(X_input)[0]

    return jsonify({"sentiment": sentiment})

# -------------------------
# Run Flask app
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
