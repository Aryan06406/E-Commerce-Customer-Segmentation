import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import joblib
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipelines.feature_eng import build_preprocessed_dataset, FEATURES

def generate_plot():
    parser = argparse.ArgumentParser(description="Visualize clusters in 2D.")
    parser.add_argument("--model", type=str, default="kmeans", choices=["kmeans", "gmm", "dbscan"])
    args = parser.parse_args()

    # 1. Load and process data
    raw_path = "data/raw/E-commerce Customer Behavior - Sheet1.csv"
    X_train, _, _, _ = build_preprocessed_dataset(raw_path)
    X_final = X_train[FEATURES]
    
    # 2. Load the Pipeline and get labels
    model_filenames = {
        "kmeans": "kmeans_customer_segmentation.pkl",
        "gmm": "gmm_customer_segmentation.pkl",
        "dbscan": "dbscan_outlier_detection.pkl"
    }
    
    pipeline = joblib.load(f"models/{model_filenames[args.model]}")
    
    if args.model == "kmeans":
        labels = pipeline.named_steps["kmeans"].predict(X_final)
    elif args.model == "gmm":
        labels = pipeline.named_steps["gaussianmixture"].predict(X_final)
    else: # dbscan
        labels = pipeline.fit_predict(X_final)
    
    # 3. Reduce to 2D using PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_final)
    
    # 4. Create the Plot
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=components[:, 0], 
        y=components[:, 1], 
        hue=labels, 
        palette='tab10', 
        s=100, 
        alpha=0.7
    )
    
    plt.title(f"Customer Segments: {args.model.upper()} (PCA Projection)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Cluster ID", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    output_path = f"data/processed/{args.model}_visualization.png"
    plt.savefig(output_path)
    print(f"{args.model.upper()} visualization saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    generate_plot()