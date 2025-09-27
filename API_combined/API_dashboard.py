from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# --------------------------
# Load trained segmentation model
# --------------------------
segmentation_model = joblib.load(
    r"D:\work\Github\Customer_segmentations\API_combined\customer_segmentation_model.pkl"
)

# Cluster label mapping
cluster_labels = {
    0: "High Spenders",
    1: "Budget-Conscious",
    2: "Trend Seekers",
    3: "Loyal Mid-Lifers",
}

DATA_PATH = r"D:\work\Github\Customer_segmentations\Simple Segmentation\marketing_campaign_clean.csv"

# --------------------------
# Descriptive profiling
# --------------------------
def descriptive_profile(cluster_label: str, has_children: int, age_group: int):
    if cluster_label == "High Spenders":
        return "Wealthy, Family-Focused" if has_children == 1 else "Wealthy, Single / No Children"
    elif cluster_label == "Budget-Conscious":
        return "Budget-Savvy Family" if has_children == 1 else "Budget-Conscious Single"
    elif cluster_label == "Trend Seekers":
        return "Young Trend Seekers" if age_group == 0 else "Trend-Seeking Adult"
    elif cluster_label == "Loyal Mid-Lifers":
        return "Loyal, Family-Oriented" if has_children == 1 else "Loyal Adult / No Children"
    else:
        return cluster_label + " - Other"

# --------------------------
# Load + Clean dataset
# --------------------------
def load_and_clean_data():
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()

    df = pd.read_csv(DATA_PATH)
    df.columns = [col.strip() for col in df.columns]

    # Remove malformed columns with commas in the name
    df = df.loc[:, ~df.columns.str.contains(",")]

    # ---- Derived Features ----
    if "Purchases" not in df.columns:
        if set(["NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]).issubset(df.columns):
            df["Purchases"] = (
                df["NumDealsPurchases"]
                + df["NumWebPurchases"]
                + df["NumCatalogPurchases"]
                + df["NumStorePurchases"]
            )

    if "Spending" not in df.columns:
        if set(["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]).issubset(df.columns):
            df["Spending"] = (
                df["MntWines"]
                + df["MntFruits"]
                + df["MntMeatProducts"]
                + df["MntFishProducts"]
                + df["MntSweetProducts"]
                + df["MntGoldProds"]
            )

    # ---- Has_Children ----
    if "Has_Children" not in df.columns:
        if "Kidhome" in df.columns and "Teenhome" in df.columns:
            df["Has_Children"] = ((df["Kidhome"] + df["Teenhome"]) > 0).astype(int)
        else:
            df["Has_Children"] = 0

    # ---- AgeGroup ----
    if "AgeGroup" not in df.columns:
        if "Year_Birth" in df.columns:
            df["AgeGroup"] = df["Year_Birth"].apply(lambda y: 0 if (2025 - y) < 35 else 1)
        else:
            df["AgeGroup"] = 1

    # ---- Ensure Response exists ----
    if "Response" not in df.columns:
        df["Response"] = 0
    else:
        df["Response"] = df["Response"].fillna(0)

    # Keep only relevant columns for model
    model_columns = ["Purchases", "Spending", "Recency", "Response", "Has_Children", "AgeGroup"]
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0  # default missing columns

    return df

# --------------------------
# Insights Endpoint
# --------------------------
@app.route("/segmentation/insights", methods=["GET"])
def segmentation_insights():
    df = load_and_clean_data()
    if df.empty:
        return jsonify({"error": "No dataset found."})

    features = ["Purchases", "Spending", "Recency", "Response"]
    df_clean = df.dropna(subset=features)

    df_clean["cluster"] = segmentation_model.predict(df_clean[features])
    df_clean["cluster_label"] = df_clean["cluster"].map(cluster_labels)

    df_clean["profile"] = df_clean.apply(
        lambda row: descriptive_profile(
            row["cluster_label"], row.get("Has_Children", 0), row.get("AgeGroup", 1)
        ),
        axis=1,
    )

    cluster_summary = (
        df_clean.groupby(["cluster_label", "profile"])
        .agg(
            count=("cluster", "size"),
            avg_purchases=("Purchases", "mean"),
            avg_spending=("Spending", "mean"),
            avg_recency=("Recency", "mean"),
        )
        .reset_index()
    )

    return jsonify({
        "total_customers": len(df_clean),
        "clusters": cluster_summary.to_dict(orient="records"),
    })

# --------------------------
# Add Customer Endpoint
# --------------------------
@app.route("/segmentation/add_customer", methods=["POST"])
def add_customer():
    data = request.get_json()

    df = load_and_clean_data()
    if df.empty:
        df = pd.DataFrame(columns=["Purchases", "Spending", "Recency", "Response", "Has_Children", "AgeGroup"])

    new_customer = pd.DataFrame([data])
    df = pd.concat([df, new_customer], ignore_index=True)

    # Save updated dataset
    df.to_csv(DATA_PATH, index=False)

    # Predict cluster for the new customer
    features = ["Purchases", "Spending", "Recency", "Response"]
    cluster_id = int(segmentation_model.predict(new_customer[features])[0])
    cluster_label = cluster_labels.get(cluster_id, "Unknown Cluster")

    profile = descriptive_profile(
        cluster_label,
        new_customer.get("Has_Children", pd.Series([0]))[0],
        new_customer.get("AgeGroup", pd.Series([1]))[0]
    )

    return jsonify({
        "message": "Customer added successfully",
        "cluster": cluster_label,
        "profile": profile
    })

# --------------------------
# Full Data Endpoint
# --------------------------
@app.route("/segmentation/full_data", methods=["GET"])
def segmentation_full_data():
    df = load_and_clean_data()
    if df.empty:
        return jsonify({"error": "No dataset found."})

    features = ["Purchases", "Spending", "Recency", "Response"]
    df_clean = df.dropna(subset=features)

    df_clean["cluster"] = segmentation_model.predict(df_clean[features])
    df_clean["cluster_label"] = df_clean["cluster"].map(cluster_labels)

    df_clean["profile"] = df_clean.apply(
        lambda row: descriptive_profile(
            row["cluster_label"], row["Has_Children"], row["AgeGroup"]
        ),
        axis=1,
    )

    return jsonify(df_clean.to_dict(orient="records"))

# --------------------------
# Run Flask app
# --------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
