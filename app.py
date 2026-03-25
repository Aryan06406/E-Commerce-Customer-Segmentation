import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "gmm_customer_segmentation.pkl")
model_pipeline = joblib.load(MODEL_PATH)

FEATURES = [
    'Age', 'Total Spend', 'Days Since Last Purchase', 'Discount Applied', 
    'Membership Type_Bronze', 'Membership Type_Gold', 'Membership Type_Silver'
]

CLUSTER_DETAILS = {
    0: {
        "name": "Deal-driven Active Customers", 
        "desc": "Active shoppers who prioritize discounts."
    },
    1: {
        "name": "Low-Spend Recent Customers", 
        "desc": "Newer users who haven't spent much yet but have visited the store recently."
    },
    2: {
        "name": "Dormant Bargain Seekers", 
        "desc": "Members who haven't shopped in a long time and only respond to heavy sales."
    },
    3: {
        "name": "High-Value Loyal Customers", 
        "desc": "The 'Gold' standard. High total spend, consistent engagement, and high loyalty."
    },
    4: {
        "name": "Churn-Risk Discount Seekers", 
        "desc": "Customers who used to spend but now only appear when there is a discount; likely to leave."
    },
    5: {
        "name": "Stable Low-Value Customers", 
        "desc": "Regular shoppers who spend small amounts consistently without much churn risk."
    },
    6: {
        "name": "At-Risk Inactive Customers", 
        "desc": "High spenders in the past who haven't returned in a very long time."
    }
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_cluster():
    try:
        data = request.get_json()
        
        processed_data = {
            "Age": float(data.get("Age", 0)),
            "Total Spend": float(data.get("Total Spend", 0)),
            "Days Since Last Purchase": float(data.get("Days Since Last Purchase", 0)),
            "Discount Applied": int(data.get("Discount Applied", 0)),
            "Membership Type_Bronze": 1 if data.get("Membership Type") == "Bronze" else 0,
            "Membership Type_Gold": 1 if data.get("Membership Type") == "Gold" else 0,
            "Membership Type_Silver": 1 if data.get("Membership Type") == "Silver" else 0
        }
        
        input_df = pd.DataFrame([processed_data])[FEATURES]
        
        cluster_id = int(model_pipeline.predict(input_df)[0])
        
        probs = model_pipeline.predict_proba(input_df)[0]
        confidence = f"{np.max(probs) * 100:.1f}%"
        
        info = CLUSTER_DETAILS.get(cluster_id, {"name": "Standard Segment", "desc": "A general customer group."})
        explanation = (
            f"With a total spend of ${processed_data['Total Spend']} and "
            f"{processed_data['Days Since Last Purchase']} days since the last purchase, "
            f"the AI identified patterns matching the '{info['name']}' segment."
        )

        return jsonify({
            "success": True,
            "cluster_id": cluster_id,
            "cluster_name": info["name"],
            "description": info["desc"],
            "explanation": explanation,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)