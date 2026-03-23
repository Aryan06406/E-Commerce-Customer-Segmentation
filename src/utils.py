"""
src/utils.py
============
Project-wide utility functions shared by pipelines, entrypoints, and tests.

Contents
--------
    Paths                     - centralised project path constants
    setup_logging()           - configure root logger
    load_raw_data()           - read raw CSV with basic validation
    save_dataframe()          - persist a DataFrame to CSV
    assert_features_present() - guard for missing model features
    cluster_size_summary()    - quick cluster distribution table
    timer()                   - context-manager for wall-clock timing
   .
"""

from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path
from typing import Iterator

import pandas as pd

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PATHS = {
    "raw_data":       PROJECT_ROOT / "data" / "raw" / "E-commerce Customer Behavior - Sheet1.csv",
    "processed_data": PROJECT_ROOT / "data" / "processed",
    "models":         PROJECT_ROOT / "models",
    "logs":           PROJECT_ROOT / "logs",
}

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(
    level: int = logging.INFO,
    log_file: str | Path | None = None,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
) -> None:
    """
    Configure the root logger with a stream handler (and optional file handler).

    Parameters
    ----------
    level    : logging level (default INFO)
    log_file : optional path to a .log file; if None, logs go to stdout only
    fmt      : log format string
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
    logging.getLogger(__name__).debug("Logging configured at level %s", logging.getLevelName(level))


# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------

def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    """
    Load the raw e-commerce CSV from *path* (or the default PATHS location).

    Raises
    ------
    FileNotFoundError  - if the file does not exist.
    ValueError         - if the file is empty.
    """
    csv_path = Path(path) if path else PATHS["raw_data"]

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: '{csv_path}'.\n"
            "Place the CSV at 'data/raw/E-commerce Customer Behavior - Sheet1.csv' "
            "or pass an explicit path."
        )

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(f"Loaded DataFrame from '{csv_path}' is empty.")

    logging.getLogger(__name__).info(
        "Loaded raw data: %d rows × %d cols from '%s'", *df.shape, csv_path
    )
    return df


def save_dataframe(
    df: pd.DataFrame,
    filename: str,
    subdir: str = "processed",
    index: bool = False,
) -> Path:
    """
    Save *df* as a CSV under ``data/<subdir>/<filename>``.

    Returns
    -------
    Path of the saved file.
    """
    out_dir = PROJECT_ROOT / "data" / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    df.to_csv(out_path, index=index)
    logging.getLogger(__name__).info("Saved DataFrame (%d rows) → '%s'", len(df), out_path)
    return out_path


# ---------------------------------------------------------------------------
# Feature validation
# ---------------------------------------------------------------------------

def assert_features_present(df: pd.DataFrame, required_features: list[str]) -> None:
    """
    Raise ValueError if any of *required_features* are missing from *df*.

    Parameters
    ----------
    df                : input DataFrame
    required_features : list of column names that must be present

    Raises
    ------
    ValueError – listing every missing column.
    """
    missing = [f for f in required_features if f not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing {len(missing)} required feature(s): {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )


# ---------------------------------------------------------------------------
# Cluster distribution helpers
# ---------------------------------------------------------------------------

def cluster_size_summary(labels, model_name: str = "") -> pd.DataFrame:
    """
    Return a tidy summary of cluster sizes and percentages.

    Parameters
    ----------
    labels     : array-like of cluster label integers
    model_name : optional label for the 'model' column

    Returns
    -------
    pd.DataFrame with columns: model, cluster, count, pct
    """
    import numpy as np

    unique, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    rows = [
        {
            "model":   model_name,
            "cluster": int(lbl),
            "count":   int(cnt),
            "pct":     round(100.0 * cnt / total, 2),
            "is_noise": lbl == -1,
        }
        for lbl, cnt in zip(unique, counts)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Timing context manager
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def timer(label: str = "block") -> Iterator[None]:
    """
    Context manager that logs wall-clock duration of the wrapped block.

    Usage
    -----
    >>> with timer("KMeans training"):
    ...     model.fit(X)
    """
    log = logging.getLogger(__name__)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        log.info("[TIMER] %s  →  %.3fs", label, elapsed)