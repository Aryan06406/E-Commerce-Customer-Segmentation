from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

MODEL_PATH = "models/gmm_customer_segmentation.pkl"
gmm_pipeline = joblib.load(MODEL_PATH)

CLUSTER_NAMES = {
    0: "Deal-driven Active Silver Customers",
    1: "Low-Spend Recent Customers",
    2: "Dormant Bronze Bargain Seekers",
    3: "High-Value Loyal Customers",
    4: "Churn-Risk Discount Seekers",
    5: "Stable Low-Value Customers",
    6: "At-Risk Inactive Customers"
}

FEATURES = ['Age', 
            'Total Spend', 
            'Days Since Last Purchase', 
            'Discount Applied', 
            'Membership Type_Bronze', 
            'Membership Type_Gold',	
            'Membership Type_Silver']

@app.route("/")
def home():
    return "Customer Segmentation API is running"

@app.route("/predict", methods=["POST"])
def predict_cluster():
    data = request.get_json()
    
    input_df = pd.DataFrame([data], columns=FEATURES)
    
    X = gmm_pipeline.named_steps['preprocess'].transform(input_df)
    cluster = int(gmm_pipeline.named_steps['gmm'].predict(X)[0])
    probs = gmm_pipeline.named_steps['gmm'].predict_proba(X)[0]

    return jsonify({
        "cluster_id": cluster,
        "cluster_name": CLUSTER_NAMES.get(cluster, "Unknown Segment"),
        "cluster_probabilities": probs.tolist()
    })

if __name__ == "__main__":
    app.run(debug=True)