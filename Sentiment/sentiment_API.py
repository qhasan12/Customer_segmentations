from flask import Flask, request, jsonify
import joblib
import re

# -------------------------
# Load model + vectorizer
# -------------------------
model, vectorizer = joblib.load(r"D:\work\Github\Customer_segmentations\Sentiment\sentiment_model.pkl")

# -------------------------
# Initialize Flask
# -------------------------
app = Flask(__name__)

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
