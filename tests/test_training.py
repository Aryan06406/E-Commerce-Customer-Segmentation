"""
tests/test_training.py
=======================
Unit and integration tests for the training & evaluation pipeline.

Tested modules
--------------
    src/pipelines/feature_eng.py
    src/pipelines/training.py
    src/pipelines/inference.py
    src/pipelines/evaluation.py
    src/utils.py

Strategy
--------
*   All tests use synthetic in-memory DataFrames that mimic the raw CSV
    schema, so no actual data files are required on disk.
*   Model-persistence tests write to a tmp_path fixture (pytest-managed
    temp dir) - no artefacts leak to the project's models/ directory.
*   Parametrised tests cover all three clustering algorithms.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline


import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipelines.evaluation import cluster_profile, evaluate_clustering
from src.pipelines.feature_eng import (
    CATEGORICAL_COLS,
    FEATURES,
    NUMERICAL_COLS,
    _fill_missing,
    build_preprocessed_dataset,
    preprocess_new_data,
)
from src.pipelines.training import train_all, train_dbscan, train_gmm, train_kmeans
from src.pipelines.inference import (
    load_pipeline,
    predict_dbscan,
    predict_gmm,
    predict_kmeans,
)
from src.utils import assert_features_present, cluster_size_summary


# ===========================================================================
# Fixtures
# ===========================================================================

N_ROWS = 200  

def _make_raw_df(n: int = N_ROWS, seed: int = 0) -> pd.DataFrame:
    """
    Synthesise a DataFrame with the same schema as the raw e-commerce CSV.
    """
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "Customer ID":             np.arange(1001, 1001 + n),
        "Age":                     rng.integers(18, 70, size=n),
        "Gender":                  rng.choice(["Male", "Female"], size=n),
        "City":                    rng.choice(["New York", "Los Angeles", "Chicago", "Houston"], size=n),
        "Membership Type":         rng.choice(["Bronze", "Silver", "Gold"], size=n),
        "Total Spend":             rng.uniform(50, 5000, size=n).round(2),
        "Items Purchased":         rng.integers(1, 50, size=n),
        "Average Rating":          rng.uniform(1, 5, size=n).round(1),
        "Discount Applied":        rng.choice([True, False], size=n),
        "Days Since Last Purchase": rng.integers(1, 365, size=n),
        "Satisfaction Level":      rng.choice(["Satisfied", "Neutral", "Unsatisfied", None], size=n),
    })
    return df


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    """Shared synthetic raw DataFrame for the module."""
    return _make_raw_df()


@pytest.fixture(scope="module")
def raw_csv(tmp_path_factory, raw_df) -> Path:
    """Write raw_df to a temporary CSV and return the path."""
    p = tmp_path_factory.mktemp("data") / "raw.csv"
    raw_df.to_csv(p, index=False)
    return p


@pytest.fixture(scope="module")
def processed(raw_csv) -> tuple:
    """Run build_preprocessed_dataset once for the whole module."""
    return build_preprocessed_dataset(raw_csv, test_size=0.2, random_state=42)


@pytest.fixture(scope="module")
def X_train(processed):
    return processed[0]


@pytest.fixture(scope="module")
def X_test(processed):
    return processed[1]


# ===========================================================================
# feature_eng tests
# ===========================================================================

class TestFillMissing:
    def test_satisfaction_level_nan_filled(self):
        df = _make_raw_df(50)
        df.loc[0:4, "Satisfaction Level"] = np.nan
        result = _fill_missing(df)
        assert result["Satisfaction Level"].isna().sum() == 0

    def test_original_not_mutated(self):
        df = _make_raw_df(30)
        df.loc[:5, "Satisfaction Level"] = np.nan
        _ = _fill_missing(df)
        assert df["Satisfaction Level"].isna().sum() > 0  # original unchanged

    def test_no_nan_passthrough(self):
        df = _make_raw_df(30)
        df["Satisfaction Level"] = "Satisfied"
        result = _fill_missing(df)
        assert result["Satisfaction Level"].isna().sum() == 0


class TestBuildPreprocessedDataset:
    def test_returns_four_objects(self, processed):
        assert len(processed) == 4

    def test_train_test_shapes(self, X_train, X_test):
        total = len(X_train) + len(X_test)
        assert len(X_test) == pytest.approx(total * 0.2, abs=2)

    def test_no_nulls_in_output(self, X_train, X_test):
        assert X_train.isna().sum().sum() == 0
        assert X_test.isna().sum().sum() == 0

    def test_customer_id_dropped(self, X_train, X_test):
        assert "Customer ID" not in X_train.columns
        assert "Customer ID" not in X_test.columns

    def test_model_features_present(self, X_train):
        for feat in FEATURES:
            assert feat in X_train.columns, f"Feature '{feat}' missing from X_train"

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_preprocessed_dataset(tmp_path / "nonexistent.csv")


class TestPreprocessNewData:
    def test_returns_dataframe(self, raw_df, processed):
        _, _, encoder, scaler = processed
        result = preprocess_new_data(raw_df.head(10), encoder, scaler)
        assert isinstance(result, pd.DataFrame)

    def test_model_features_present(self, raw_df, processed):
        _, _, encoder, scaler = processed
        result = preprocess_new_data(raw_df.head(5), encoder, scaler)
        assert_features_present(result, FEATURES)


# ===========================================================================
# training tests
# ===========================================================================

class TestTrainKMeans:
    def test_returns_pipeline_and_labels(self, X_train, tmp_path):
        pipeline, labels = train_kmeans(X_train, model_dir=tmp_path)
        assert isinstance(pipeline, Pipeline)
        assert len(labels) == len(X_train)

    def test_correct_number_of_clusters(self, X_train, tmp_path):
        _, labels = train_kmeans(X_train, model_dir=tmp_path)
        assert len(np.unique(labels)) == 7

    def test_pkl_created(self, X_train, tmp_path):
        train_kmeans(X_train, model_dir=tmp_path)
        assert (tmp_path / "kmeans_customer_segmentation.pkl").exists()

    def test_no_noise_labels(self, X_train, tmp_path):
        _, labels = train_kmeans(X_train, model_dir=tmp_path)
        assert -1 not in labels


class TestTrainGMM:
    def test_returns_pipeline_and_labels(self, X_train, tmp_path):
        pipeline, labels = train_gmm(X_train, model_dir=tmp_path)
        assert isinstance(pipeline, Pipeline)
        assert len(labels) == len(X_train)

    def test_correct_number_of_components(self, X_train, tmp_path):
        _, labels = train_gmm(X_train, model_dir=tmp_path)
        assert len(np.unique(labels)) <= 7

    def test_pkl_created(self, X_train, tmp_path):
        train_gmm(X_train, model_dir=tmp_path)
        assert (tmp_path / "gmm_customer_segmentation.pkl").exists()


class TestTrainDBSCAN:
    def test_returns_pipeline_and_labels(self, X_train, tmp_path):
        pipeline, labels = train_dbscan(X_train, model_dir=tmp_path)
        assert isinstance(pipeline, Pipeline)
        assert len(labels) == len(X_train)

    def test_pkl_created(self, X_train, tmp_path):
        train_dbscan(X_train, model_dir=tmp_path)
        assert (tmp_path / "dbscan_outlier_detection.pkl").exists()

    def test_noise_label_possible(self, X_train, tmp_path):
        """DBSCAN is allowed to produce noise labels (−1); test just verifies dtype."""
        _, labels = train_dbscan(X_train, model_dir=tmp_path)
        assert labels.dtype in (np.int32, np.int64, int)


class TestTrainAll:
    def test_returns_dict_with_three_keys(self, X_train, tmp_path):
        results = train_all(X_train, model_dir=tmp_path)
        assert set(results.keys()) == {"kmeans", "gmm", "dbscan"}

    def test_all_pkls_exist(self, X_train, tmp_path):
        train_all(X_train, model_dir=tmp_path)
        expected = [
            "kmeans_customer_segmentation.pkl",
            "gmm_customer_segmentation.pkl",
            "dbscan_outlier_detection.pkl",
        ]
        for fname in expected:
            assert (tmp_path / fname).exists(), f"{fname} not created"


# ===========================================================================
# inference tests
# ===========================================================================

class TestLoadPipeline:
    def test_loads_existing_pkl(self, X_train, tmp_path):
        train_kmeans(X_train, model_dir=tmp_path)
        pipeline = load_pipeline(tmp_path / "kmeans_customer_segmentation.pkl")
        assert isinstance(pipeline, Pipeline)

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pipeline(tmp_path / "nonexistent.pkl")


@pytest.mark.parametrize("trainer,predictor,fname", [
    (train_kmeans, predict_kmeans, "kmeans_customer_segmentation.pkl"),
    (train_gmm,    predict_gmm,    "gmm_customer_segmentation.pkl"),
])
class TestPredict:
    def test_label_count_matches_input(self, trainer, predictor, fname, X_train, X_test, tmp_path):
        trainer(X_train, model_dir=tmp_path)
        labels = predictor(X_test, model_dir=tmp_path)
        assert len(labels) == len(X_test)

    def test_labels_are_integers(self, trainer, predictor, fname, X_train, X_test, tmp_path):
        trainer(X_train, model_dir=tmp_path)
        labels = predictor(X_test, model_dir=tmp_path)
        assert np.issubdtype(labels.dtype, np.integer)

    def test_missing_feature_raises(self, trainer, predictor, fname, X_train, X_test, tmp_path):
        trainer(X_train, model_dir=tmp_path)
        X_bad = X_test.drop(columns=[FEATURES[0]])
        with pytest.raises(ValueError, match="missing required features"):
            predictor(X_bad, model_dir=tmp_path)


# ===========================================================================
# evaluation tests
# ===========================================================================

class TestEvaluateClustering:
    def _dummy_data(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((100, 7))
        labels = np.repeat([0, 1, 2, 3], 25)
        return X, labels

    def test_returns_three_metrics(self):
        X, labels = self._dummy_data()
        result = evaluate_clustering(X, labels)
        assert "silhouette" in result
        assert "davies_bouldin" in result
        assert "calinski_harabasz" in result

    def test_prefix_applied(self):
        X, labels = self._dummy_data()
        result = evaluate_clustering(X, labels, prefix="train_")
        assert all(k.startswith("train_") for k in result)

    def test_handles_noise_labels(self):
        X, labels = self._dummy_data()
        labels_with_noise = labels.copy()
        labels_with_noise[:10] = -1     
        result = evaluate_clustering(X, labels_with_noise)
        assert result["silhouette"] is not None

    def test_returns_none_for_single_cluster(self):
        X = np.random.randn(50, 3)
        labels = np.zeros(50, dtype=int) 
        result = evaluate_clustering(X, labels)
        assert result["silhouette"] is None


class TestClusterProfile:
    def test_returns_dataframe(self, X_train, tmp_path):
        _, labels = train_kmeans(X_train, model_dir=tmp_path)
        profile = cluster_profile(X_train, labels)
        assert isinstance(profile, pd.DataFrame)

    def test_excludes_noise_rows(self, X_train, tmp_path):
        _, labels = train_dbscan(X_train, model_dir=tmp_path)
        profile = cluster_profile(X_train, labels)
        assert -1 not in profile.index


# ===========================================================================
# utils tests
# ===========================================================================

class TestClusterSizeSummary:
    def test_totals_match_input(self):
        labels = np.array([0, 0, 1, 1, 1, 2])
        df = cluster_size_summary(labels)
        assert df["count"].sum() == len(labels)

    def test_noise_label_flagged(self):
        labels = np.array([-1, -1, 0, 0, 1])
        df = cluster_size_summary(labels)
        noise_rows = df[df["cluster"] == -1]
        assert len(noise_rows) == 1
        assert bool(noise_rows["is_noise"].iloc[0]) is True

    def test_pct_sums_to_100(self):
        labels = np.array([0, 0, 1, 2])
        df = cluster_size_summary(labels)
        assert abs(df["pct"].sum() - 100.0) < 1e-6


class TestAssertFeaturesPresent:
    def test_passes_when_all_present(self):
        df = pd.DataFrame(columns=FEATURES + ["extra"])
        assert_features_present(df, FEATURES)  # no exception

    def test_raises_when_missing(self):
        df = pd.DataFrame(columns=FEATURES[:-1])  # last feature missing
        with pytest.raises(ValueError, match="missing"):
            assert_features_present(df, FEATURES)