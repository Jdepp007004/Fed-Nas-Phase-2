"""
client/data_loader.py
M4: TCGA dataset ingestion, preprocessing, and DataLoader creation.
Owner: Nikhil Garuda
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from sklearn.preprocessing import LabelEncoder, MinMaxScaler  # noqa: E402
from sklearn.model_selection import StratifiedShuffleSplit  # noqa: E402

from shared.model_schema import (  # noqa: E402
    REQUIRED_COLUMNS,
    TARGET_COLUMNS,
    FEATURE_RANGES,
    CATEGORICAL_VALUES,
    MIN_SAMPLES,
    INPUT_DIM,
)


# ─── Custom Exception ────────────────────────────────────────────────────────

class SchemaValidationError(Exception):
    """Raised when a client CSV fails schema validation during loading."""


# ─── Step 1: Load & Clean ────────────────────────────────────────────────────

def load_tcga_dataset(csv_path: str, schema: dict) -> pd.DataFrame:
    """
    Load and clean the TCGA Clinical Subset CSV.

    Parameters
    ----------
    csv_path : str  — absolute path to the local TCGA CSV file
    schema   : dict — required schema from the project (from list_projects/join_project)

    Returns
    -------
    pd.DataFrame — cleaned, schema-filtered dataframe

    Raises
    ------
    SchemaValidationError  if minimum requirements are not met
    FileNotFoundError      if csv_path does not exist
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    # Normalise column names: strip whitespace, lower-case
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    required = schema.get("required_columns", REQUIRED_COLUMNS)
    min_samples = schema.get("min_samples", MIN_SAMPLES)

    # Check required columns present
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise SchemaValidationError(
            f"Missing required columns: {missing_cols}"
        )

    # Filter to schema columns only
    available = [c for c in required if c in df.columns]
    df = df[available].copy()

    # Drop rows where > 30% of values are missing
    threshold = 0.3 * len(df.columns)
    df = df.dropna(thresh=int(len(df.columns) - threshold))

    # Fill remaining NaNs: median for numeric, mode for categorical
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode_val = df[col].mode()
                df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else "unknown")

    if len(df) < min_samples:
        raise SchemaValidationError(
            f"Dataset has {len(df)} samples but project requires at least {min_samples}."
        )

    return df


# ─── Step 2: Feature Engineering ─────────────────────────────────────────────

def preprocess_features(df: pd.DataFrame, schema: dict) -> tuple:
    """
    Transform the clean dataframe into model-ready numpy arrays.

    Parameters
    ----------
    df     : pd.DataFrame — cleaned output from load_tcga_dataset()
    schema : dict         — server-provided schema with feature_ranges, target_columns, encoding_maps

    Returns
    -------
    tuple: (X: np.ndarray of shape (N, INPUT_DIM), y: dict)
    """
    target_cols = schema.get("target_columns", TARGET_COLUMNS)
    feature_ranges = schema.get("feature_ranges", FEATURE_RANGES)
    cat_values = schema.get("categorical_values", CATEGORICAL_VALUES)

    df = df.copy()

    # ── Extract targets before touching features ─────────────────────────────
    y_reg = _extract_regression_target(df, target_cols["regression"])
    y_tox = _extract_toxicity_target(df, target_cols["toxicity"])
    y_bin = _extract_binary_target(df, target_cols["binary"])

    # Drop target columns from feature matrix
    feature_df = df.drop(
        columns=[v for v in target_cols.values() if v in df.columns],
        errors="ignore",
    )

    # ── Label-encode low-cardinality categoricals ────────────────────────────
    le = LabelEncoder()
    encoded_parts = []
    num_cols = []

    for col in feature_df.columns:
        if col in cat_values or (not pd.api.types.is_numeric_dtype(feature_df[col])):
            # Label encode
            feature_df[col] = feature_df[col].astype(str).str.lower().str.strip()
            encoded_col = le.fit_transform(feature_df[col])
            encoded_parts.append(encoded_col.reshape(-1, 1))
        else:
            num_cols.append(col)

    # ── Min-max normalise numerical columns ──────────────────────────────────
    if num_cols:
        scaler = MinMaxScaler()  # noqa: F841
        num_array = feature_df[num_cols].values.astype(np.float32)
        # Use schema-defined ranges if available, else fit from data
        for idx, col in enumerate(num_cols):
            if col in feature_ranges:
                lo, hi = feature_ranges[col]
                num_array[:, idx] = np.clip(
                    (num_array[:, idx] - lo) / max(hi - lo, 1e-8), 0.0, 1.0
                )
        encoded_parts.append(num_array)

    if encoded_parts:
        X_raw = np.hstack(encoded_parts).astype(np.float32)
    else:
        X_raw = np.zeros((len(df), 1), dtype=np.float32)

    # ── Pad or truncate to INPUT_DIM ─────────────────────────────────────────
    N, F = X_raw.shape
    if F < INPUT_DIM:
        X = np.zeros((N, INPUT_DIM), dtype=np.float32)
        X[:, :F] = X_raw
    else:
        X = X_raw[:, :INPUT_DIM]

    y = {
        "regression": y_reg.astype(np.float32),
        "toxicity": y_tox.astype(np.int64),
        "binary": y_bin.astype(np.float32),
    }

    return X, y


# ─── Target Extraction Helpers ────────────────────────────────────────────────

def _extract_regression_target(df, col):
    if col in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).values
    else:
        vals = np.zeros(len(df), dtype=np.float32)
    return vals.astype(np.float32)


def _extract_toxicity_target(df, col):
    """Map treatment_outcome to integer toxicity grade 0–3."""
    if col in df.columns:
        mapping = {
            "complete response": 0,
            "partial response": 1,
            "stable disease": 2,
            "progressive disease": 3,
        }
        vals = df[col].astype(str).str.lower().str.strip().map(mapping)
        vals = vals.fillna(0).astype(np.int64).values
    else:
        vals = np.zeros(len(df), dtype=np.int64)
    return vals


def _extract_binary_target(df, col):
    """Map vital_status to 0 (alive) / 1 (dead)."""
    if col in df.columns:
        vals = df[col].astype(str).str.lower().str.strip()
        vals = (vals == "dead").astype(np.float32).values
    else:
        vals = np.zeros(len(df), dtype=np.float32)
    return vals


# ─── Step 3: DataLoader Creation ─────────────────────────────────────────────

def create_federated_dataloader(
    X: np.ndarray,
    y: dict,
    split: float = 0.2,
    batch_size: int = 32,
) -> tuple:
    """
    Wrap preprocessed arrays into train/val PyTorch DataLoaders.

    Parameters
    ----------
    X          : np.ndarray  — feature matrix (N, INPUT_DIM)
    y          : dict        — {regression, toxicity, binary} arrays
    split      : float       — validation fraction (default 0.2)
    batch_size : int         — training batch size (default 32)

    Returns
    -------
    tuple: (train_loader: DataLoader, val_loader: DataLoader)
    """
    N = len(X)  # noqa: F841
    y_bin = y["binary"].astype(np.int32)

    # Stratified split on binary target
    sss = StratifiedShuffleSplit(n_splits=1, test_size=split, random_state=42)
    for train_idx, val_idx in sss.split(X, y_bin):
        pass

    def make_ds(idx):
        return TensorDataset(
            torch.from_numpy(X[idx]),
            torch.from_numpy(y["regression"][idx]),
            torch.from_numpy(y["toxicity"][idx]),
            torch.from_numpy(y["binary"][idx]),
        )

    train_ds = make_ds(train_idx)
    val_ds = make_ds(val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader


# ─── Convenience Pipeline ─────────────────────────────────────────────────────

def build_dataloaders_from_csv(
    csv_path: str,
    schema: dict,
    split: float = 0.2,
    batch_size: int = 32,
) -> tuple:
    """
    Full pipeline: CSV → clean → preprocess → DataLoaders.

    Returns
    -------
    tuple: (train_loader, val_loader)
    """
    df = load_tcga_dataset(csv_path, schema)
    X, y = preprocess_features(df, schema)
    return create_federated_dataloader(X, y, split, batch_size)
