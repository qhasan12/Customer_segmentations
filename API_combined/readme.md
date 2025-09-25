#### Customer Analytics API 🚀

## An AI-driven Customer Insights and Engagement Platform that provides:
    -Customer Segmentation
    -Churn Prediction
    -Product Recommendations
    -Sentiment Analysis

## 🔗 Live API

**Deployed on**: https://your-app-name.ondigitalocean.app

## 📋 API Endpoints
#### 1. Customer Churn Prediction

```bash
POST /predict-churn
Content-Type: application/json

Request Body:
{
    "recency": 10,
    "frequency": 5,
    "monetary": 100,
    "avg_payment_value": 50,
    "avg_review_score": 4.0
}

Response:
{
    "churn_prediction": 1,
    "churn_probability": 0.82
}

#### 2. Product Recommendations

POST /recommend
Content-Type: application/json

Request Body:
{
    "cart": ["milk", "bread"]
}

Response:
    {
        "product_recommendations": [
        {
            "item": "butter",
            "reason": "Because you bought milk, customers also often buy butter",
            "confidence": 0.75,
            "lift": 1.2
        }
        ],
        "aisle_recommendations": [
        {
            "item": "cheese",
            "reason": "Because you bought bread, customers also often buy cheese",
            "confidence": 0.65,
            "lift": 1.1
        }
        ],
        "department_recommendations": [
            {
            "item": "dairy",
            "reason": "Because you bought milk, customers also often buy dairy products",
            "confidence": 0.70,
            "lift": 1.3
            }
        ]
    }

#### 3. Sentiment Analysis

POST /sentiment
Content-Type: application/json

    Request Body:
    {
        "text": "I love this product! It's amazing."
    }

    Response:
    {
        "sentiment": "positive"
    }

#### 4. Customer Segmentation

POST /predict-segment
Content-Type: application/json

Request Body:
[
    {
        "AgeGroup": 2,
        "Education_Encoded": 1,
        "Marital_Status": 0,
        "Income": 50000,
        "Has_Children": 1,
        "Purchases": 20,
        "Spending": 1500,
        "Recency": 5,
        "Response": 1
    }
]

Response:
[
    {
        "AgeGroup": 2,
        "Education_Encoded": 1,
        "Marital_Status": 0,
        "Income": 50000,
        "Has_Children": 1,
        "Purchases": 20,
        "Spending": 1500,
        "Recency": 5,
        "Response": 1,
        "Predicted_Cluster": 0,
        "Cluster_Label": "High Spenders"
    }
]

5. Root Endpoint

GET /

Response:
{
"message": "Unified Customer Analytics API is running 🚀",
    "endpoints": [
        "/predict-churn",
        "/recommend",
        "/sentiment",
        "/predict-segment"
    ]
}