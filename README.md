This updated `README.md` incorporates the new **Advanced Inference** workflow, highlights the mandatory data flags, and documents the new model artifacts (`encoder.pkl` and `scaler.pkl`) generated during training.

---

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
Train the models and save the feature engineering artifacts (`encoder.pkl`, `scaler.pkl`). **Note:** You must provide the data path explicitly.
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
| Model       | Use Case             | Characteristics                                        |
| :---------: | :------------------: | :----------------------------------------------------: |
| **KMeans**  | General Segmentation | Efficient, assumes spherical clusters.                 |
| **GMM**     | Soft Clustering      | Handles overlapping segments using probability.        |
| **DBSCAN**  | Outlier Detection    | Identifies "noise" (Cluster -1) and non-linear shapes. |

## ⚠️ Troubleshooting & Notes
* **Mandatory Arguments:** Ensure you provide the `--data` flag for training and the `--input` flag for inference. Omitting these will result in a `TypeError`.
* **Artifact Dependency:** `inference.py` requires `encoder.pkl` and `scaler.pkl` to exist in the `models/` folder. If they are missing, re-run `train.py`.
* **Feature Consistency:** The models are trained on 7 specific features: `Age`, `Total Spend`, `Days Since Last Purchase`, `Discount Applied`, and 3 `Membership Type` categories.
* **Windows Warnings:** `UserWarning` regarding physical cores (Loky) can be safely ignored.