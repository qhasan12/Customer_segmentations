from flask import Flask, request, jsonify
import joblib
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import re
import numpy as np

# ------------------------
# Load trained model + tokenizer + label encoder
# ------------------------
clf, tokenizer, le = joblib.load(r"D:\work\Github\Customer_segmentations\Sentiment\sentiment_model.pkl")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

app = Flask(__name__)

# ------------------------
# Function to get BERT embeddings
# ------------------------
def get_bert_embedding(text):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# ------------------------
# Prediction route (multiple texts)
# ------------------------
@app.route("/sentiment", methods=["POST"])
def predict_sentiment():
    try:
        data = request.get_json()
        texts = data.get("texts", [])  # Expect a list of texts

        if not texts or not isinstance(texts, list):
            return jsonify({"error": "Please provide a list of texts"}), 400

        results = []
        for text in texts:
            # Preprocess
            cleaned = re.sub(r'[^a-zA-Z\s]', '', text.lower())

            # Get embedding
            emb = get_bert_embedding(cleaned).reshape(1, -1)

            # Predict
            pred = clf.predict(emb)[0]
            sentiment = le.inverse_transform([pred])[0]

            results.append({"text": text, "sentiment": sentiment})

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------
# Health check
# ------------------------
@app.route("/", methods=["GET"])
def home():
    return "BERT Sentiment Analysis API is running 🚀"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
