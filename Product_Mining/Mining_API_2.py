from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load rules at startup
rules_product = joblib.load(r"D:\work\Github\Customer_segmentations\Product_Mining\rules_product.pkl")
rules_aisle = joblib.load(r"D:\work\Github\Customer_segmentations\Product_Mining\rules_aisle.pkl")
rules_department = joblib.load(r"D:\work\Github\Customer_segmentations\Product_Mining\rules_department.pkl")


def recommend(cart_items, rules, top_n=5):
    cart_items = set(cart_items)
    recs = []
    for _, row in rules.iterrows():
        if row['antecedents'].issubset(cart_items):
            for consequent in row['consequents']:
                if consequent not in cart_items:
                    reason = (f"Because you bought {', '.join(row['antecedents'])}, "
                              f"customers also often buy {consequent}")
                    recs.append({
                        "item": consequent,
                        "reason": reason,
                        "confidence": float(row['confidence']),
                        "lift": float(row['lift'])
                    })
    # sort by confidence & lift
    recs = sorted(recs, key=lambda x: (x['confidence'], x['lift']), reverse=True)
    # keep top_n unique
    seen, final = set(), []
    for r in recs:
        if r["item"] not in seen:
            final.append(r)
            seen.add(r["item"])
        if len(final) >= top_n:
            break
    return final


@app.route("/recommend-product", methods=["POST"])
def recommend_product():
    data = request.json
    cart_items = data.get("cart", [])
    recs = recommend(cart_items, rules_product)
    return jsonify(recs)


@app.route("/recommend-aisle", methods=["POST"])
def recommend_aisle():
    data = request.json
    cart_items = data.get("cart", [])
    recs = recommend(cart_items, rules_aisle)
    return jsonify(recs)


@app.route("/recommend-department", methods=["POST"])
def recommend_department():
    data = request.json
    cart_items = data.get("cart", [])
    recs = recommend(cart_items, rules_department)
    return jsonify(recs)


if __name__ == "__main__":
    app.run(debug=True)
