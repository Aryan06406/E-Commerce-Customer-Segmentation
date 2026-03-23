"""
entrypoint/inference.py
========================
CLI entry point for running inference on new data using saved pipelines.

Usage
-----
    # run all three models on a CSV and print predictions
    python entrypoint/inference.py --input data/new_customers.csv

    # run only KMeans and save output to a file
    python entrypoint/inference.py --input data/new_customers.csv --models kmeans --output results/

    # specify a custom model directory
    python entrypoint/inference.py --input data/new.csv --model-dir models/ --models gmm

Output
------
For each requested model, a CSV is written to --output (default: results/) with:
    - all original columns
    - <model>_cluster column appended

Exit codes
----------
    0  – success
    1  – file / path error
    2  – inference error
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.pipelines.feature_eng import FEATURES, preprocess_new_data
from src.pipelines.inference import predict_all, predict_dbscan, predict_gmm, predict_kmeans
from src.utils import assert_features_present, cluster_size_summary, setup_logging, timer

logger = logging.getLogger(__name__)

_PREDICTORS = {
    "kmeans": predict_kmeans,
    "gmm":    predict_gmm,
    "dbscan": predict_dbscan,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="inference.py",
        description="Run cluster inference on new e-commerce customer data.",
    )
    p.add_argument(
        "--input",
        required=True,
        help="Path to input CSV (raw or pre-encoded schema matching training data).",
    )
    p.add_argument(
        "--model-dir",
        default="models",
        help="Directory containing the .pkl pipeline files (default: models/)",
    )
    p.add_argument(
        "--models",
        nargs="+",
        choices=list(_PREDICTORS.keys()),
        default=list(_PREDICTORS.keys()),
        help="Which models to run inference with (default: all three)",
    )
    p.add_argument(
        "--output",
        default="results",
        help="Directory to write output CSVs (default: results/)",
    )
    p.add_argument(
        "--preprocessed",
        action="store_true",
        help=(
            "Flag to indicate the input CSV is already preprocessed "
            "(OHE + scaled). If omitted, the raw schema is assumed and "
            "basic imputation is applied."
        ),
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


def _load_input(path: str | Path, preprocessed: bool) -> pd.DataFrame:
    """Load input CSV and validate it contains the required model features."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: '{path}'")

    df = pd.read_csv(path)
    logger.info("Loaded input: %d rows × %d cols from '%s'", *df.shape, path)

    if preprocessed:
        assert_features_present(df, FEATURES)
    # If not preprocessed, feature_eng.preprocess_new_data will handle it
    return df


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(level=getattr(logging, args.log_level))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)

    # ------------------------------------------------------------------
    # 1. Load input
    # ------------------------------------------------------------------
    try:
        df_raw = _load_input(args.input, preprocessed=args.preprocessed)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    # ------------------------------------------------------------------
    # 2. Optional light preprocessing for raw inputs
    # ------------------------------------------------------------------
    if not args.preprocessed:
        logger.info("Applying lightweight preprocessing (imputation only) to raw input.")
        # For a proper production system, the encoder & scaler should be
        # persisted alongside models.  Here we apply only the imputation
        # step and leave OHE / scaling to the pipeline's internal scaler.
        from src.pipelines.feature_eng import _fill_missing
        df_raw = _fill_missing(df_raw)

    # ------------------------------------------------------------------
    # 3. Run inference
    # ------------------------------------------------------------------
    results: dict = {}
    try:
        for model_name in args.models:
            predictor = _PREDICTORS[model_name]
            with timer(f"{model_name} inference"):
                labels = predictor(df_raw, model_dir=model_dir)
            results[model_name] = labels
            summary = cluster_size_summary(labels, model_name=model_name)
            logger.info("\n%s predictions:\n%s", model_name, summary.to_string(index=False))
    except FileNotFoundError as exc:
        logger.error("Model file missing: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Inference failed: %s", exc)
        return 2

    # ------------------------------------------------------------------
    # 4. Save outputs
    # ------------------------------------------------------------------
    for model_name, labels in results.items():
        df_out = df_raw.copy()
        df_out[f"{model_name}_cluster"] = labels

        out_path = out_dir / f"predictions_{model_name}.csv"
        df_out.to_csv(out_path, index=False)
        logger.info("Saved predictions → '%s'", out_path)

    logger.info("Inference complete. Results written to '%s'.", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())