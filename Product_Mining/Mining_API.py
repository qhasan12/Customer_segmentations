from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS


app = Flask(__name__)
CORS(app)  # allow all origins by default

# === Load pre-trained rules (exported from notebook with joblib.dump) ===
rules = joblib.load(r"D:\work\Github\Customer_segmentations\Product_Mining\rules.pkl")

# === Recommendation function (directly inside API) ===
def recommend_products(cart_items, rules, top_n=5, min_conf=0.0, min_lift=0.0):
    cart_set = set(cart_items)

    # Filter strong rules
    filtered_rules = rules[
        (rules['confidence'] >= min_conf) &
        (rules['lift'] >= min_lift)
    ]

    # Match rules where any antecedent item is in the cart
    matched_rules = filtered_rules[filtered_rules['antecedents'].apply(lambda x: len(x & cart_set) > 0)]

    recommendations = []
    for _, row in matched_rules.iterrows():
        for consequent in row['consequents']:
            if consequent not in cart_set:
                recommendations.append({
                    "item": consequent,
                    "confidence": float(row['confidence']),
                    "lift": float(row['lift'])
                })

    # Sort + deduplicate
    recommendations = sorted(recommendations, key=lambda x: (x['confidence'], x['lift']), reverse=True)
    seen, final_recs = set(), []
    for rec in recommendations:
        if rec["item"] not in seen:
            final_recs.append(rec)
            seen.add(rec["item"])
        if len(final_recs) >= top_n:
            break

    return final_recs

# === API endpoint ===
@app.route("/recommend", methods=["POST"])
def recommend_api():
    data = request.get_json()
    cart = data.get("cart", [])
    recs = recommend_products(cart, rules, top_n=5)
    return jsonify({"cart": cart, "recommendations": recs})

if __name__ == "__main__":
    app.run(debug=True)
