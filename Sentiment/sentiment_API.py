from flask import Flask, request, jsonify
import joblib

# Load model and vectorizer together
model, vectorizer = joblib.load(r"D:\work\Github\Customer_segmentations\Sentiment\sentiment_model.pkl")

# Initialize Flask
app = Flask(__name__)

@app.route("/sentiment", methods=["POST"])
def predict():
    data = request.get_json(force=True)   # get JSON input
    text = data.get("text", "")           # extract text from input

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Transform text using loaded vectorizer
    X = vectorizer.transform([text])

    # Prediction
    prediction = model.predict(X)[0]

    return jsonify({"sentiment": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
