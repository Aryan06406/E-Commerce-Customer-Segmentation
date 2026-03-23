"""
src/pipelines/training.py
=========================
Builds and persists the three clustering pipelines from notebook 3:

    KMeans  (n_clusters=7,  random_state=42, n_init=10)
    GMM     (n_components=7, covariance_type='full', random_state=42)
    DBSCAN  (eps=0.8, min_samples=5)

Each pipeline is:
    Pipeline([('preprocess', StandardScaler()), ('<model>', <estimator>)])

Saved artefacts
---------------
    models/kmeans_customer_segmentation.pkl
    models/gmm_customer_segmentation.pkl
    models/dbscan_outlier_detection.pkl

Public API
----------
    train_kmeans(X_train, model_dir) -> (pipeline, labels)
    train_gmm(X_train, model_dir)    -> (pipeline, labels)
    train_dbscan(X_train, model_dir) -> (pipeline, labels)
    train_all(X_train, model_dir)    -> dict[str, (pipeline, labels)]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .feature_eng import FEATURES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyper-parameter constants (tuned from notebook elbow / BIC analysis)
# ---------------------------------------------------------------------------

KMEANS_PARAMS: dict = dict(n_clusters=7, random_state=42, n_init=10)
GMM_PARAMS: dict    = dict(n_components=7, covariance_type="full", random_state=42)
DBSCAN_PARAMS: dict = dict(eps=0.8, min_samples=5)

MODEL_DIR_DEFAULT = Path(__file__).resolve().parents[2] / "models"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_scaler_pipeline(estimator) -> Pipeline:
    """Wrap an estimator with a StandardScaler pre-processing step."""
    return Pipeline([
        ("preprocess", StandardScaler()),
        (type(estimator).__name__.lower(), estimator),
    ])


def _save(pipeline: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info("Saved pipeline → %s", path)


def _get_labels_from_pipeline(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Retrieve cluster labels from a fitted pipeline.
    GMM's pipeline step is named 'gaussianmixture'; others expose fit_predict.
    """
    step_name = pipeline.steps[-1][0]
    estimator = pipeline.named_steps[step_name]

    if isinstance(estimator, GaussianMixture):
        # GMM doesn't support fit_predict on the pipeline level
        X_scaled = pipeline.named_steps["preprocess"].transform(X)
        return estimator.predict(X_scaled)

    return pipeline.fit_predict(X)


# ---------------------------------------------------------------------------
# Public trainers
# ---------------------------------------------------------------------------

def train_kmeans(
    X_train: pd.DataFrame,
    model_dir: str | Path = MODEL_DIR_DEFAULT,
) -> Tuple[Pipeline, np.ndarray]:
    """
    Train a KMeans clustering pipeline on the 7-feature subset and save it.

    Parameters
    ----------
    X_train   : preprocessed training DataFrame (output of feature_eng)
    model_dir : directory to persist the .pkl artefact

    Returns
    -------
    (kmeans_pipeline, labels)
        labels – cluster assignments for X_train[FEATURES]
    """
    X = X_train[FEATURES]
    logger.info("Training KMeans  params=%s  n_samples=%d", KMEANS_PARAMS, len(X))

    pipeline = _make_scaler_pipeline(KMeans(**KMEANS_PARAMS))
    labels   = pipeline.fit_predict(X)

    _save(pipeline, Path(model_dir) / "kmeans_customer_segmentation.pkl")
    logger.info("KMeans training complete.  Unique clusters: %s", np.unique(labels))
    return pipeline, labels


def train_gmm(
    X_train: pd.DataFrame,
    model_dir: str | Path = MODEL_DIR_DEFAULT,
) -> Tuple[Pipeline, np.ndarray]:
    """
    Train a Gaussian Mixture Model pipeline on the 7-feature subset and save it.

    Returns
    -------
    (gmm_pipeline, labels)
        labels – soft-assignment (predict) cluster labels for X_train[FEATURES]
    """
    X = X_train[FEATURES]
    logger.info("Training GMM  params=%s  n_samples=%d", GMM_PARAMS, len(X))

    pipeline = _make_scaler_pipeline(GaussianMixture(**GMM_PARAMS))
    pipeline.fit(X)

    # GMM on pipeline level: scale first, then predict from fitted GMM step
    X_scaled = pipeline.named_steps["preprocess"].transform(X)
    labels   = pipeline.named_steps["gaussianmixture"].predict(X_scaled)

    _save(pipeline, Path(model_dir) / "gmm_customer_segmentation.pkl")
    logger.info("GMM training complete.  Unique clusters: %s", np.unique(labels))
    return pipeline, labels


def train_dbscan(
    X_train: pd.DataFrame,
    model_dir: str | Path = MODEL_DIR_DEFAULT,
) -> Tuple[Pipeline, np.ndarray]:
    """
    Train a DBSCAN outlier-detection pipeline on the 7-feature subset and save it.

    Notes
    -----
    DBSCAN label ``-1`` indicates noise / outlier points.

    Returns
    -------
    (dbscan_pipeline, labels)
    """
    X = X_train[FEATURES]
    logger.info("Training DBSCAN  params=%s  n_samples=%d", DBSCAN_PARAMS, len(X))

    pipeline = _make_scaler_pipeline(DBSCAN(**DBSCAN_PARAMS))
    labels   = pipeline.fit_predict(X)

    n_outliers = (labels == -1).sum()
    logger.info(
        "DBSCAN training complete.  Clusters: %d  |  Outliers: %d",
        len(set(labels) - {-1}),
        n_outliers,
    )

    _save(pipeline, Path(model_dir) / "dbscan_outlier_detection.pkl")
    return pipeline, labels


def train_all(
    X_train: pd.DataFrame,
    model_dir: str | Path = MODEL_DIR_DEFAULT,
) -> Dict[str, Tuple[Pipeline, np.ndarray]]:
    """
    Convenience wrapper – trains all three pipelines in sequence.

    Returns
    -------
    {
        'kmeans': (pipeline, labels),
        'gmm':    (pipeline, labels),
        'dbscan': (pipeline, labels),
    }
    """
    model_dir = Path(model_dir)
    results: Dict[str, Tuple[Pipeline, np.ndarray]] = {}

    results["kmeans"] = train_kmeans(X_train, model_dir)
    results["gmm"]    = train_gmm(X_train, model_dir)
    results["dbscan"] = train_dbscan(X_train, model_dir)

    logger.info("All three pipelines trained and saved to '%s'.", model_dir)
    return results