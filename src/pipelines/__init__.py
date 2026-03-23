"""

=============
src/pipelines
=============

Exposes the four pipeline modules for the e-commerce customer-segmentation
project:

    feature_eng  - raw-data cleaning & feature engineering
    training     - build and persist KMeans / GMM / DBSCAN pipelines
    inference    - load saved pipelines and produce cluster assignments
    evaluation   - clustering quality metrics
    
"""

from .feature_eng import build_preprocessed_dataset, FEATURES, CATEGORICAL_COLS, NUMERICAL_COLS
from .training   import train_all, train_kmeans, train_gmm, train_dbscan
from .inference  import load_pipeline, predict_kmeans, predict_gmm, predict_dbscan
from .evaluation import evaluate_clustering, full_report

__all__ = [
    # feature_eng
    "build_preprocessed_dataset",
    "FEATURES",
    "CATEGORICAL_COLS",
    "NUMERICAL_COLS",
    # training
    "train_all",
    "train_kmeans",
    "train_gmm",
    "train_dbscan",
    # inference
    "load_pipeline",
    "predict_kmeans",
    "predict_gmm",
    "predict_dbscan",
    # evaluation
    "evaluate_clustering",
    "full_report",
]