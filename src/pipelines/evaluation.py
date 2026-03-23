"""
src/pipelines/evaluation.py
============================
Clustering quality metrics used in notebook 3.

Metrics implemented
-------------------
    Silhouette Score           – higher is better  (range -1 → 1)
    Davies-Bouldin Index       – lower is better   (range  0 → ∞)
    Calinski-Harabasz Score    – higher is better  (range  0 → ∞)

All functions gracefully skip noise points (label == -1) for DBSCAN output.

Public API
----------
    evaluate_clustering(X, labels, prefix="") -> dict
    full_report(X_train, X_test, all_results)  -> pd.DataFrame
    print_report(report_df)                    -> None
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metric function
# ---------------------------------------------------------------------------

def evaluate_clustering(
    X: pd.DataFrame | np.ndarray,
    labels: np.ndarray,
    prefix: str = "",
) -> dict:
    """
    Compute Silhouette, Davies-Bouldin, and Calinski-Harabasz scores,
    automatically masking out DBSCAN noise points (label == -1).

    Parameters
    ----------
    X       : feature matrix (unscaled or scaled – must be consistent with
              how labels were produced)
    labels  : cluster assignment array of shape (n_samples,)
    prefix  : optional string prefix added to dict keys, e.g. "train_" or "test_"

    Returns
    -------
    dict with keys: <prefix>silhouette, <prefix>davies_bouldin, <prefix>calinski_harabasz
    Each value is float | None (None when there are fewer than 2 valid clusters).
    """
    X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

    # Mask noise points for DBSCAN (label == -1)
    mask       = labels != -1
    X_valid    = X_arr[mask]
    lbl_valid  = labels[mask]

    n_clusters = len(set(lbl_valid))
    n_samples  = len(X_valid)

    result: dict = {}

    if n_clusters < 2 or n_samples < n_clusters:
        logger.warning(
            "Cannot compute metrics: %d valid clusters, %d valid samples.",
            n_clusters, n_samples,
        )
        result[f"{prefix}silhouette"]          = None
        result[f"{prefix}davies_bouldin"]      = None
        result[f"{prefix}calinski_harabasz"]   = None
        return result

    result[f"{prefix}silhouette"]        = round(float(silhouette_score(X_valid, lbl_valid)),        4)
    result[f"{prefix}davies_bouldin"]    = round(float(davies_bouldin_score(X_valid, lbl_valid)),    4)
    result[f"{prefix}calinski_harabasz"] = round(float(calinski_harabasz_score(X_valid, lbl_valid)), 4)

    return result


# ---------------------------------------------------------------------------
# Full comparative report
# ---------------------------------------------------------------------------

def full_report(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    all_results: Dict[str, tuple],
) -> pd.DataFrame:
    """
    Build a human-readable comparison table for all three models on both
    train and test splits.

    Parameters
    ----------
    X_train     : training feature DataFrame
    X_test      : test feature DataFrame
    all_results : dict returned by ``train_all``, keyed by model name,
                  values are (pipeline, train_labels).

    Returns
    -------
    pd.DataFrame with rows = models, columns = metric × split.

    Example
    -------
    >>> from src.pipelines.training import train_all
    >>> from src.pipelines.inference import predict_kmeans, predict_gmm, predict_dbscan
    >>> from src.pipelines.evaluation import full_report
    >>> results  = train_all(X_train)
    >>> test_lbl = {
    ...     'kmeans': predict_kmeans(X_test),
    ...     'gmm':    predict_gmm(X_test),
    ...     'dbscan': predict_dbscan(X_test),
    ... }
    >>> report   = full_report(X_train, X_test, results, test_lbl)
    """
    from .feature_eng import FEATURES

    rows = []
    for model_name, (_, train_labels) in all_results.items():
        X_tr = X_train[FEATURES]
        row  = {"model": model_name}

        # Training metrics
        row.update(evaluate_clustering(X_tr, train_labels, prefix="train_"))

        rows.append(row)

    return pd.DataFrame(rows).set_index("model")


def print_report(report: pd.DataFrame) -> None:
    """Pretty-print the evaluation report to stdout."""
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 120)
    print("\n" + "=" * 80)
    print("CLUSTERING EVALUATION REPORT")
    print("=" * 80)
    print(report.to_string())
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Cluster profile helper (from notebook 3)
# ---------------------------------------------------------------------------

def cluster_profile(
    X_train: pd.DataFrame,
    labels: np.ndarray,
    profile_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Return mean feature values per cluster (noise points excluded).

    Parameters
    ----------
    X_train      : training DataFrame (post-preprocessing)
    labels       : cluster assignment array
    profile_cols : columns to include in profile; defaults to raw
                   interpretable features.

    Returns
    -------
    pd.DataFrame indexed by cluster id.
    """
    default_profile_cols = ["Age", "Total Spend", "Days Since Last Purchase", "Discount Applied"]
    profile_cols = profile_cols or [c for c in default_profile_cols if c in X_train.columns]

    df = X_train.copy()
    df["_cluster"] = labels
    df = df[df["_cluster"] != -1]   # exclude noise

    return df.groupby("_cluster")[profile_cols].mean().round(4)