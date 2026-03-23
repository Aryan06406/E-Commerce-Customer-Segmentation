"""
src/pipelines/inference.py
==========================
Loads persisted pipelines and produces cluster / segment assignments
for new (or held-out) data.

Public API
----------
    load_pipeline(path)                     -> sklearn Pipeline
    predict_kmeans(X, model_dir)            -> np.ndarray of cluster labels
    predict_gmm(X, model_dir)               -> np.ndarray of cluster labels
    predict_dbscan(X, model_dir)            -> np.ndarray of cluster labels (-1 = outlier)
    predict_all(X, model_dir)               -> dict[str, np.ndarray]
    predict_gmm_proba(X, model_dir)         -> np.ndarray of shape (n_samples, n_components)

Notes
-----
*   All predict_* functions accept a DataFrame with either the raw 
    7-feature subset (already preprocessed) or a wider DataFrame from
    which FEATURES are extracted automatically.
*   The pipelines carry their own StandardScaler as the first step, so do not re-scale before calling these functions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline

from .feature_eng import FEATURES

logger = logging.getLogger(__name__)

MODEL_DIR_DEFAULT = Path(__file__).resolve().parents[2] / "models"

_MODEL_FILES: dict[str, str] = {
    "kmeans": "kmeans_customer_segmentation.pkl",
    "gmm":    "gmm_customer_segmentation.pkl",
    "dbscan": "dbscan_outlier_detection.pkl",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pipeline(path: str | Path) -> Pipeline:
    """
    Load and return a persisted sklearn Pipeline from *path*.

    Raises
    ------
    FileNotFoundError - if the .pkl file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: '{path}'.  "
            "Run entrypoint/train.py first to generate it."
        )
    pipeline = joblib.load(path)
    logger.info("Loaded pipeline from '%s'", path)
    return pipeline


def _extract_features(X: pd.DataFrame) -> pd.DataFrame:
    """Return only the 7-feature model subset from X, if the full column set is present."""
    missing = [f for f in FEATURES if f not in X.columns]
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required features: {missing}.  "
            f"Expected: {FEATURES}"
        )
    return X[FEATURES]


def _model_path(model_key: str, model_dir: Path) -> Path:
    return model_dir / _MODEL_FILES[model_key]


# ---------------------------------------------------------------------------
# Public predict functions
# ---------------------------------------------------------------------------

def predict_kmeans(
    X: pd.DataFrame,
    model_dir: str | Path = MODEL_DIR_DEFAULT,
) -> np.ndarray:
    """
    Assign KMeans cluster labels to *X*.

    Parameters
    ----------
    X         : DataFrame that includes the 7 FEATURES columns
    model_dir : directory containing the .pkl files.

    Returns
    -------
    np.ndarray of int cluster labels, shape (n_samples,)
    """
    pipeline = load_pipeline(_model_path("kmeans", Path(model_dir)))
    X_feat   = _extract_features(X)
    labels   = pipeline.predict(X_feat)
    logger.debug("KMeans predicted %d labels, unique=%s", len(labels), np.unique(labels))
    return labels


def predict_gmm(
    X: pd.DataFrame,
    model_dir: str | Path = MODEL_DIR_DEFAULT,
) -> np.ndarray:
    """
    Assign GMM cluster labels to *X*.

    Returns
    -------
    np.ndarray of int cluster labels, shape (n_samples,)
    """
    pipeline = load_pipeline(_model_path("gmm", Path(model_dir)))
    X_feat   = _extract_features(X)

    # GMM lives inside a pipeline; scale then call the GMM's predict
    X_scaled = pipeline.named_steps["preprocess"].transform(X_feat)
    gmm: GaussianMixture = pipeline.named_steps["gaussianmixture"]
    labels   = gmm.predict(X_scaled)

    logger.debug("GMM predicted %d labels, unique=%s", len(labels), np.unique(labels))
    return labels


def predict_gmm_proba(
    X: pd.DataFrame,
    model_dir: str | Path = MODEL_DIR_DEFAULT,
) -> np.ndarray:
    """
    Return soft cluster-membership probabilities from the GMM.

    Returns
    -------
    np.ndarray of shape (n_samples, n_components) - each row sums to 1.
    """
    pipeline = load_pipeline(_model_path("gmm", Path(model_dir)))
    X_feat   = _extract_features(X)
    X_scaled = pipeline.named_steps["preprocess"].transform(X_feat)
    gmm: GaussianMixture = pipeline.named_steps["gaussianmixture"]
    return gmm.predict_proba(X_scaled)


def predict_dbscan(
    X: pd.DataFrame,
    model_dir: str | Path = MODEL_DIR_DEFAULT,
) -> np.ndarray:
    """
    Assign DBSCAN cluster / outlier labels to *X*.

    Notes
    -----
    Label ``-1`` means the point is classified as **noise / outlier**.

    Returns
    -------
    np.ndarray of int labels, shape (n_samples,)
    """
    pipeline = load_pipeline(_model_path("dbscan", Path(model_dir)))
    X_feat   = _extract_features(X)
    labels   = pipeline.fit_predict(X_feat)   # DBSCAN has no standalone predict()

    n_outliers = (labels == -1).sum()
    logger.debug(
        "DBSCAN assigned %d labels.  Outliers: %d  Clusters: %d",
        len(labels), n_outliers, len(set(labels) - {-1}),
    )
    return labels


def predict_all(
    X: pd.DataFrame,
    model_dir: str | Path = MODEL_DIR_DEFAULT,
) -> Dict[str, np.ndarray]:
    """
    Run all three models and return a dict of label arrays.

    Returns
    -------
    {
        'kmeans': np.ndarray,
        'gmm':    np.ndarray,
        'dbscan': np.ndarray,
    }
    """
    model_dir = Path(model_dir)
    return {
        "kmeans": predict_kmeans(X, model_dir),
        "gmm":    predict_gmm(X, model_dir),
        "dbscan": predict_dbscan(X, model_dir),
    }