"""
client/schema_validator.py
M4: Validates client CSV dataframe against the project's required schema.
Owner: Nikhil Garuda
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from collections import namedtuple  # noqa: E402
import pandas as pd  # noqa: E402

from shared.model_schema import MIN_SAMPLES  # noqa: E402

# ─── Result Type ─────────────────────────────────────────────────────────────

ValidationResult = namedtuple("ValidationResult", ["passed", "errors", "warnings"])


# ─── Main Validator ──────────────────────────────────────────────────────────

def validate_schema(df: pd.DataFrame, expected_schema: dict) -> ValidationResult:
    """
    Perform a comprehensive multi-check validation of the client dataframe.

    Checks performed:
      1. All required columns are present
      2. Column data types match expected types
      3. Numerical column values are within expected bounds
      4. Categorical columns contain only known categories
      5. Row count >= min_samples

    Parameters
    ----------
    df              : pd.DataFrame — the client's loaded data
    expected_schema : dict — schema from the server project listing

    Returns
    -------
    ValidationResult(passed: bool, errors: list[str], warnings: list[str])
    """
    errors: list = []
    warnings: list = []

    required_columns = expected_schema.get("required_columns", [])
    column_types = expected_schema.get("column_types", {})
    value_bounds = expected_schema.get("feature_ranges", {})
    cat_values = expected_schema.get("categorical_values", {})
    min_samples = expected_schema.get("min_samples", MIN_SAMPLES)

    # Normalise column names
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # ── Check 1: Required columns present ───────────────────────────────────
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        errors.append(f"Missing {len(missing)} required column(s): {missing[:10]}{'...' if len(missing)>10 else ''}")  # noqa: E225, E501

    present = [c for c in required_columns if c in df.columns]  # noqa: F841

    # ── Check 2: Data type compatibility ─────────────────────────────────────
    for col, expected_type in column_types.items():
        if col not in df.columns:
            continue
        if expected_type in ("float", "int"):
            if not pd.api.types.is_numeric_dtype(df[col]):
                coerced = pd.to_numeric(df[col], errors="coerce")
                nan_pct = coerced.isna().mean()
                if nan_pct > 0.5:
                    errors.append(
                        f"Column '{col}' expected numeric but {nan_pct:.0%} values cannot be converted."
                    )
                else:
                    warnings.append(
                        f"Column '{col}' has {nan_pct:.0%} non-numeric values that will be filled with median."
                    )

    # ── Check 3: Value range bounds ───────────────────────────────────────────
    for col, (lo, hi) in value_bounds.items():
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(numeric) == 0:
            continue
        out_of_range = ((numeric < lo) | (numeric > hi)).mean()
        if out_of_range > 0.05:
            warnings.append(
                f"Column '{col}': {out_of_range:.0%} of values outside expected range [{lo}, {hi}]."
            )

    # ── Check 4: Categorical values ────────────────────────────────────────────
    for col, known_cats in cat_values.items():
        if col not in df.columns:
            continue
        known_set = {str(c).lower() for c in known_cats}
        actual_vals = df[col].dropna().astype(str).str.lower().str.strip().unique()
        unknown = [v for v in actual_vals if v not in known_set]
        if unknown:
            warnings.append(
                f"Column '{col}' contains {len(unknown)} unknown category value(s): {unknown[:5]}"
            )

    # ── Check 5: Minimum row count ────────────────────────────────────────────
    if len(df) < min_samples:
        errors.append(
            f"Dataset has only {len(df)} rows; project requires at least {min_samples}."
        )

    passed = len(errors) == 0
    return ValidationResult(passed=passed, errors=errors, warnings=warnings)


# ─── Convenience: display results ─────────────────────────────────────────────

def format_validation_report(result: ValidationResult) -> str:
    """Format a ValidationResult into a human-readable string for the UI."""
    lines = []
    status = "✅ PASSED" if result.passed else "❌ FAILED"
    lines.append(f"Schema Validation: {status}")
    if result.errors:
        lines.append("\nErrors (must fix before joining):")
        for e in result.errors:
            lines.append(f"  • {e}")
    if result.warnings:
        lines.append("\nWarnings (will be handled automatically):")
        for w in result.warnings:
            lines.append(f"  ⚠ {w}")
    return "\n".join(lines)
