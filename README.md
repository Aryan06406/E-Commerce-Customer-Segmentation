# E-Commerce Customer Segmentation

An end-to-end Machine Learning pipeline for segmenting e-commerce customers using **KMeans**, **Gaussian Mixture Models (GMM)**, and **DBSCAN**. This project features a modular architecture designed for easy transitions between training and production-style batch inference.

## 🚀 Project Overview
This repository provides a system to analyze customer behavior and categorize users into distinct segments based on purchasing patterns, demographics, and engagement metrics.

* **Clustering Models:** KMeans (Centroid-based), GMM (Probabilistic), and DBSCAN (Density-based).
* **Production Inference:** Dedicated logic to handle raw data by persisting encoders and scalers alongside models.
* **Dimensionality Reduction:** PCA-based 2D visualization for interpreting high-dimensional clusters.

## 📂 Project Structure
```text
.
├── data/
│   ├── raw/                # Original E-commerce CSV files
│   └── processed/          # Cleaned data and visualization exports
├── entrypoint/
│   ├── train.py            # Model training & artifact generation
│   ├── inference.py        # Advanced Batch Inference (Raw data -> Results)
│   ├── predict.py          # Basic prediction CLI
│   └── visualize.py        # PCA visualization CLI
├── models/                 # Serialized .pkl (Models, Encoders, Scalers)
├── results/                # Output folder for batch inference CSVs
├── src/
│   ├── pipelines/          # Core logic (Feature Eng, Training, Inference)
│   └── utils.py            # Logging, timers, and shared helpers
├── tests/                  # Pytest suite
├── requirements.txt        # Project dependencies
└── README.md
```

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/E-Commerce-Customer-Segmentation.git
    cd E-Commerce-Customer-Segmentation
    ```

2.  **Set up a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Usage Guide

The project follows a linear pipeline: **Train → Inference → Visualize**.

### 1. Training & Artifact Generation
Train the models and save the feature engineering artifacts (`preprocessor_encoder.pkl`, `preprocessor_scaler.pkl`). 
**Note:** You must provide the data path explicitly.
```bash
python entrypoint/train.py --data "data/raw/E-commerce Customer Behavior - Sheet1.csv"
```
* **Output:** Generates 5 `.pkl` files in `models/` (3 models + encoder + scaler).
* **Metrics:** Displays Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Score.

### 2. Advanced Batch Inference
Apply all trained models to new raw data. This script automatically handles encoding and scaling using the saved artifacts.
```bash
python entrypoint/inference.py --input "data/raw/E-commerce Customer Behavior - Sheet1.csv" --output results/
```
* **Behavior:** Runs inference across all models by default.
* **Output:** Individual CSVs saved to `results/` with original data and appended cluster labels.

### 3. PCA Visualization
Project high-dimensional clusters into 2D space for visualization.
```bash
# Options for --model: kmeans, gmm, dbscan
python entrypoint/visualize.py --model kmeans
```
* **Output:** Generates a PNG plot in `data/processed/`.

## 🧪 Testing
The project includes a comprehensive test suite covering preprocessing and model persistence.
```bash
pytest tests/
```

## 📊 Model Comparison Details

### Feature Sets Used

Two feature configurations were evaluated across all models:

| Feature | Set Features | Count |
| **FS-1** (Core) | `Age`, `Total Spend`, `Days Since Last Purchase`, `Discount Applied`, `Membership Type_Bronze`, `Membership Type_Gold`, `Membership Type_Silver` | 7 |
| **FS-2** (Extended) | All of FS-1 **+** `Satisfaction Level_Neutral`, `Satisfaction Level_Satisfied`, `Satisfaction Level_Unsatisfied`, `Items Purchased`, `Average Rating` | 12 |

### Results Comparison: Feature Set 1 vs Feature Set 2

> **Metrics:** Silhouette Score ↑ (higher = better), Davies-Bouldin Index ↓ (lower = better).  
> DBSCAN does not support `predict()` on held-out data, so test scores are not applicable (N/A).

| Model      | Feature Set     | Clusters | Sil. Score (Train) | Sil. Score (Test) | DB Index (Train) | DB Index (Test) |
| :--------: | :-------------: | :------: | :----------------: | :---------------: | :--------------: | :-------------: |
| **KMeans** | FS-1 (Core)     |     7    | 0.8199             |       0.8195      |      0.2579      |      0.2547     |
| **KMeans** | FS-2 (Extended) |     8    | 0.7251             |       0.7651      |      0.4148      |      0.2873     |
| **GMM**    | FS-1 (Core)     |     7    | 0.8199             |       0.8195      |      0.2579      |      0.2547     |
| **GMM**    | FS-2 (Extended) |     7    | 0.7872             |       0.7838      |      0.3097      |      0.3103     |
| **DBSCAN** | FS-1 (Core)     |   Auto   | 0.8116             |        N/A        |      0.8733      |       N/A       |
| **DBSCAN** | FS-2 (Extended) |   Auto   | 0.7163             |        N/A        |      1.1423      |       N/A       |

### Key Takeaways

* **FS-1 consistently outperforms FS-2** on Silhouette Score across all three models, suggesting the 7 core features produce tighter, more well-separated clusters.
* **KMeans & GMM** deliver identical scores on FS-1, both achieving the highest Silhouette (0.8199 train / 0.8195 test) and lowest Davies-Bouldin (0.2579 / 0.2547) — indicating GMM converges to the same hard partition as KMeans on this feature space.
* **GMM on FS-2** improves meaningfully over KMeans on FS-2 (Silhouette 0.787 vs 0.725 train), showing GMM's probabilistic assignment handles the richer feature space better.
* **DBSCAN** achieves a competitive Silhouette on FS-1 (0.8116) but has a noticeably high Davies-Bouldin (0.8733), reflecting its noise cluster (label `-1`) distorting inter-cluster distance metrics. Its performance degrades further on FS-2.
* **Recommendation:** Use **FS-1 with KMeans or GMM** for the most reliable production segmentation. Reserve DBSCAN for outlier/anomaly detection use cases regardless of feature set.

---

### Model Characteristics
| Model       | Use Case             | Characteristics                                        |
| :---------: | :------------------: | :----------------------------------------------------: |
| **KMeans**  | General Segmentation | Efficient, assumes spherical clusters.                 |
| **GMM**     | Soft Clustering      | Handles overlapping segments using probability.        |
| **DBSCAN**  | Outlier Detection    | Identifies "noise" (Cluster -1) and non-linear shapes. |

## ⚠️ Troubleshooting & Notes
* **Mandatory Arguments:** Ensure you provide the `--data` flag for training and the `--input` flag for inference. Omitting these will result in a `TypeError`.
* **Artifact Dependency:** `inference.py` requires `preprocessor_encoder.pkl` and `preprocessor_scaler.pkl` to exist in the `models/` folder.
* **Feature Consistency:** The models are trained on 7 specific features: `Age`, `Total Spend`, `Days Since Last Purchase`, `Discount Applied`, and 3 `Membership Type` categories.
* **Windows Warnings:** `UserWarning` regarding physical cores (Loky) can be safely ignored.