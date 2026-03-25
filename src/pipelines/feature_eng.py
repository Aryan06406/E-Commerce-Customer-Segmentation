"""
src/pipelines/feature_eng.py
============================
Feature engineering pipeline that mirrors the EDA + preprocessing steps
performed in notebooks 1__eda.ipynb and 3__training.ipynb.

Steps
-----
1.  Load raw CSV.
2.  Fill missing 'Satisfaction Level' with mode  (notebook 1).
3.  Assert no remaining critical nulls.
4.  Separate numerical / categorical columns.
5.  One-hot encode categorical columns.
6.  Standard-scale numerical columns.
7.  Drop 'Customer ID' (non-feature identifier column).
8.  Return a train/test split ready for clustering.

Exported symbols
----------------
    CATEGORICAL_COLS        – categorical column names in raw data
    NUMERICAL_COLS          – numerical column names in raw data
    FEATURES                – the 7 feature subset used by all models
    build_preprocessed_dataset(raw_path, test_size, random_state)
        -> (X_train, X_test, encoder, scaler)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column-level constants (derived from notebook exploration)
# ---------------------------------------------------------------------------

CATEGORICAL_COLS: list[str] = [
    "Membership Type"
]

# Includes bool columns – sklearn's StandardScaler handles bool/int64/float64
NUMERICAL_COLS: list[str] = [
    "Age",
    "Days Since Last Purchase",
    "Total Spend",
    "Discount Applied",   # bool in raw CSV → treated as 0/1 after cast
]

# The 7-feature subset selected after PCA/UMAP exploration (notebook 3)
FEATURES: list[str] = [
    "Age",
    "Total Spend",
    "Days Since Last Purchase",
    "Discount Applied",
    "Membership Type_Bronze",
    "Membership Type_Gold",
    "Membership Type_Silver",
]

# Columns to drop before modelling (identifiers, non-informative)
_DROP_COLS: list[str] = ["Customer ID", 'Satisfaction Level', 'Items Purchased', 'Average Rating']


# ---------------------------------------------------------------------------
# EDA helpers
# ---------------------------------------------------------------------------

def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce the EDA null-filling from notebook 1:
      - 'Satisfaction Level': fill NaN with column mode.
    Returns a copy so the original DataFrame is unchanged.
    """
    df = df.copy()
    if df["Satisfaction Level"].isna().any():
        mode_val = df["Satisfaction Level"].mode()[0]
        df["Satisfaction Level"] = df["Satisfaction Level"].fillna(mode_val)
        logger.info("Filled 'Satisfaction Level' NaN with mode='%s'", mode_val)
    return df


def _assert_no_nulls(df: pd.DataFrame, strict_cols: list[str] | None = None) -> None:
    """Raise ValueError if any column (or a given subset) still contains NaN."""
    cols = strict_cols or df.columns.tolist()
    null_counts = df[cols].isna().sum()
    bad = null_counts[null_counts > 0]
    if not bad.empty:
        raise ValueError(f"Unexpected nulls after EDA step:\n{bad}")


# ---------------------------------------------------------------------------
# Encoding + scaling
# ---------------------------------------------------------------------------

def _encode_categorical(
    df: pd.DataFrame,
    cat_cols: list[str],
    encoder: OneHotEncoder | None = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, OneHotEncoder]:
    """
    One-hot encode *cat_cols* and append the resulting columns to *df*,
    dropping the originals.  Returns (transformed_df, fitted_encoder).

    If *encoder* is supplied and *fit=False*, the existing encoder is used
    (transform-only mode for test data / inference).
    """
    df = df.copy()
    if fit:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoder.fit(df[cat_cols])
        logger.debug("OneHotEncoder fitted on %d categorical columns.", len(cat_cols))

    if encoder is None:
        raise ValueError("encoder must be provided when fit=False.")

    encoded_arr = encoder.transform(df[cat_cols])
    encoded_cols = list(encoder.get_feature_names_out(cat_cols))
    df[encoded_cols] = encoded_arr
    df = df.drop(columns=cat_cols)
    return df, encoder


def _scale_numerical(
    df: pd.DataFrame,
    num_cols: list[str],
    scaler: StandardScaler | None = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Standard-scale *num_cols* in-place (on a copy).
    Returns (transformed_df, fitted_scaler).
    """
    df = df.copy()
    if fit:
        scaler = StandardScaler()
        scaler.fit(df[num_cols])
        logger.debug("StandardScaler fitted on %d numerical columns.", len(num_cols))

    if scaler is None:
        raise ValueError("scaler must be provided when fit=False.")

    df[num_cols] = scaler.transform(df[num_cols])
    return df, scaler


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_preprocessed_dataset(
    raw_path: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder, StandardScaler]:
    """
    Full preprocessing pipeline from raw CSV to model-ready train/test splits.

    Parameters
    ----------
    raw_path     : path to the raw CSV (e.g. data/raw/E-commerce Customer Behavior - Sheet1.csv)
    test_size    : fraction reserved for test split (default 0.20)
    random_state : random seed for reproducibility

    Returns
    -------
    X_train  : pd.DataFrame   -  preprocessed training features (all columns)
    X_test   : pd.DataFrame   - preprocessed test features (all columns)
    encoder  : OneHotEncoder  - fitted on training data
    scaler   : StandardScaler - fitted on training data

    Notes
    -----
    Both X_train and X_test contain **all** post-encoding columns.
    To use only the 7-feature model subset, index with ``FEATURES``:
        X_train[FEATURES], X_test[FEATURES]
    """
    raw_path = Path(raw_path)
    logger.info("Loading raw data from '%s'", raw_path)
    df = pd.read_csv(raw_path)
    logger.info("Raw data shape: %s", df.shape)

    # ---- EDA ---------------------------------------------------------------
    df = _fill_missing(df)
    _assert_no_nulls(df)

    # ---- Encode + scale ----------------------------------------------------
    # Identify which num cols are present 
    present_num_cols = [c for c in NUMERICAL_COLS if c in df.columns]
    present_cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]

    df, encoder = _encode_categorical(df, present_cat_cols, fit=True)
    df, scaler  = _scale_numerical(df, [c for c in present_num_cols if c in df.columns], fit=True)

    # ---- Drop non-feature identifier columns --------------------------------
    drop_existing = [c for c in _DROP_COLS if c in df.columns]
    if drop_existing:
        df = df.drop(columns=drop_existing)
        logger.debug("Dropped identifier columns: %s", drop_existing)

    # ---- Train / test split -------------------------------------------------
    X_train, X_test = train_test_split(df, test_size=test_size, random_state=random_state)
    logger.info("Train size: %d  |  Test size: %d", len(X_train), len(X_test))

    return X_train, X_test, encoder, scaler


def preprocess_new_data(
    df_new: pd.DataFrame,
    encoder: OneHotEncoder,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """
    Apply a *fitted* encoder + scaler to unseen data for inference.

    Parameters
    ----------
    df_new  : raw DataFrame with the same schema as training data
    encoder : fitted OneHotEncoder from ``build_preprocessed_dataset``
    scaler  : fitted StandardScaler from ``build_preprocessed_dataset``

    Returns
    -------
    Preprocessed DataFrame with the same column layout as training output.
    """
    df = df_new.copy()

    # Fill missing the same way as training
    df = _fill_missing(df)

    present_cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    present_num_cols = [c for c in NUMERICAL_COLS if c in df.columns]

    df, _ = _encode_categorical(df, present_cat_cols, encoder=encoder, fit=False)
    df, _ = _scale_numerical(df, [c for c in present_num_cols if c in df.columns],
                              scaler=scaler, fit=False)

    drop_existing = [c for c in _DROP_COLS if c in df.columns]
    if drop_existing:
        df = df.drop(columns=drop_existing)

    return df