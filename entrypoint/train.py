"""
entrypoint/train.py
====================
CLI entry point for end-to-end training.

Usage
-----
    # train all three models (default)
    python entrypoint/train.py

    # specify a custom data path and model output dir
    python entrypoint/train.py --data data/raw/my_data.csv --model-dir models/

    # train only specific models
    python entrypoint/train.py --models kmeans gmm

    # adjust train/test split
    python entrypoint/train.py --test-size 0.15

Pipeline
--------
    1.  Load & preprocess raw data (feature_eng)
    2.  Train requested models     (training)
    3.  Evaluate on train split    (evaluation)
    4.  Print cluster size summary (utils)
    5.  Save all artefacts to models/

Exit codes
----------
    0  – success
    1  – data / file error
    2  – training error
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipelines.evaluation import cluster_profile, evaluate_clustering, print_report
from src.pipelines.feature_eng import FEATURES, build_preprocessed_dataset
from src.pipelines.training import train_all, train_dbscan, train_gmm, train_kmeans
from src.utils import cluster_size_summary, setup_logging, timer

logger = logging.getLogger(__name__)

_TRAINERS = {
    "kmeans": train_kmeans,
    "gmm":    train_gmm,
    "dbscan": train_dbscan,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="train.py",
        description="Train e-commerce customer segmentation clustering models.",
    )
    p.add_argument(
        "--data",
        default=None,
        help="Path to the raw CSV.  Defaults to data/raw/E-commerce Customer Behavior - Sheet1.csv",
    )
    p.add_argument(
        "--model-dir",
        default="models",
        help="Directory to save .pkl artefacts (default: models/)",
    )
    p.add_argument(
        "--models",
        nargs="+",
        choices=list(_TRAINERS.keys()),
        default=list(_TRAINERS.keys()),
        help="Which models to train (default: all three)",
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data held out for evaluation (default: 0.20)",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    p.add_argument(
        "--log-file",
        default=None,
        help="Optional path to write logs to a file.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(level=getattr(logging, args.log_level), log_file=args.log_file)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load & preprocess
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1 – Feature Engineering")
    logger.info("=" * 60)

    try:
        with timer("Feature engineering"):
            X_train, X_test, encoder, scaler = build_preprocessed_dataset(
                raw_path=args.data,
                test_size=args.test_size,
                random_state=args.random_state,
            )
    except FileNotFoundError as exc:
        logger.error("Data file not found: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Feature engineering failed: %s", exc)
        return 1

    logger.info("X_train: %s  |  X_test: %s", X_train.shape, X_test.shape)

    # Persist encoder + scaler so inference can fully reproduce preprocessing
    import joblib
    joblib.dump(encoder, model_dir / "preprocessor_encoder.pkl")
    joblib.dump(scaler,  model_dir / "preprocessor_scaler.pkl")
    logger.info("Saved encoder → models/preprocessor_encoder.pkl")
    logger.info("Saved scaler  → models/preprocessor_scaler.pkl")

    # ------------------------------------------------------------------
    # 2. Train
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 2 – Model Training  (models: %s)", args.models)
    logger.info("=" * 60)

    results: dict = {}
    try:
        for model_name in args.models:
            trainer = _TRAINERS[model_name]
            with timer(f"{model_name} training"):
                pipeline, labels = trainer(X_train, model_dir=model_dir)
            results[model_name] = (pipeline, labels)
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        return 2

    # ------------------------------------------------------------------
    # 3. Evaluate
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 3 – Evaluation")
    logger.info("=" * 60)

    import pandas as pd

    eval_rows = []
    for model_name, (_, labels) in results.items():
        X_feat = X_train[FEATURES]
        metrics = evaluate_clustering(X_feat.values, labels, prefix="train_")
        metrics["model"] = model_name
        eval_rows.append(metrics)

        summary = cluster_size_summary(labels, model_name=model_name)
        logger.info("\n%s cluster sizes:\n%s", model_name, summary.to_string(index=False))

    eval_df = pd.DataFrame(eval_rows).set_index("model")
    print_report(eval_df)

    # ------------------------------------------------------------------
    # 4. Cluster profiles (interpretability)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 4 – Cluster Profiles")
    logger.info("=" * 60)

    for model_name, (_, labels) in results.items():
        profile = cluster_profile(X_train, labels)
        logger.info("\n%s cluster profile:\n%s", model_name.upper(), profile.to_string())

    logger.info("Training complete.  Artefacts saved in '%s'.", model_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())