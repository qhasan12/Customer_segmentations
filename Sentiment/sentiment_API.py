from flask import Flask, request, jsonify
import joblib
import re
import os
from flask_cors import CORS

# -------------------------
# Load model + vectorizer
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.pkl")

model, vectorizer = joblib.load(MODEL_PATH)

# -------------------------
# Initialize Flask
# -------------------------
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# -------------------------
# Predict single text
# -------------------------
@app.route("/sentiment", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    cleaned = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    X_input = vectorizer.transform([cleaned])
    sentiment = model.predict(X_input)[0]

    return jsonify({"sentiment": sentiment})


# -------------------------
# Predict multiple texts
# -------------------------
@app.route("/sentiments", methods=["POST"])
def predict_batch():
    data = request.get_json(force=True)
    texts = data.get("texts", [])

    if not texts or not isinstance(texts, list):
        return jsonify({"error": "Please provide a list of texts"}), 400

    results = []
    for t in texts:
        cleaned = re.sub(r'[^a-zA-Z\s]', '', t.lower())
        X_input = vectorizer.transform([cleaned])
        sentiment = model.predict(X_input)[0]
        results.append({"text": t, "sentiment": sentiment})

    return jsonify({"results": results})


# -------------------------
# Predefined grocery test cases
# -------------------------
GROCERY_TESTS = [
    "Fresh strawberries were perfect today, super sweet and juicy.",
    "Half the shelves were empty again, what’s going on?",
    "Cashier was polite, but the line took forever.",
    "The meat smelled off… not buying here again.",
    "Love how they finally added oat milk to the dairy section.",
    "Prices are crazy high lately — feels like robbery.",
    "Store was clean and well organized, easy to find everything.",
    "Bananas were green yesterday, brown today… how?",
    "Self-checkout is faster, but the machines freeze too often.",
    "The bakery bread was warm, soft, and honestly the best part of my day.",
    "Parking lot was a nightmare, almost gave up before getting inside.",
    "Thank you for restocking the snacks my kids love.",
    "Produce section looked messy and half the veggies were wilted.",
    "Customer service desk actually solved my issue in minutes, shockingly good.",
    "Sale signs are confusing — ended up paying more than expected."
]

@app.route("/test-sentiments", methods=["GET"])
def test_sentiments():
    results = []
    for t in GROCERY_TESTS:
        cleaned = re.sub(r'[^a-zA-Z\s]', '', t.lower())
        X_input = vectorizer.transform([cleaned])
        sentiment = model.predict(X_input)[0]
        results.append({"text": t, "sentiment": sentiment})

    return jsonify({"results": results})


# -------------------------
# Run Flask app
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
