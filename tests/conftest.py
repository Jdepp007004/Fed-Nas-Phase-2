"""
tests/conftest.py
Shared pytest fixtures for the FL Platform test suite.
"""

import os
import sys
import json
import tempfile
import numpy as np
import pandas as pd
import pytest

# ── Path setup: make shared, server, client importable ───────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in [ROOT, os.path.join(ROOT, "server"), os.path.join(ROOT, "client")]:
    if sub not in sys.path:
        sys.path.insert(0, sub)

# ── Set dummy encryption key before any imports ───────────────────────────────
import base64
os.environ["FL_ENCRYPTION_KEY"] = base64.b64encode(b"test_key_32bytes_padding_000000!").decode()
os.environ["JWT_SECRET"] = "test_jwt_secret"


# ─── Synthetic TCGA CSV ───────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def tcga_csv_path(tmp_path_factory):
    """Create a small synthetic TCGA CSV with required columns."""
    from shared.model_schema import REQUIRED_COLUMNS

    rng = np.random.default_rng(42)
    n = 200

    data = {}
    for col in REQUIRED_COLUMNS:
        # Numeric columns get random floats; categorical get simple strings
        if col in (
            "age_at_diagnosis", "days_to_death", "days_to_last_follow_up",
            "tumor_largest_dimension", "weight", "bmi", "height",
            "number_of_cycles", "year_of_diagnosis", "overall_survival",
            "days_to_treatment_start", "days_to_treatment_end",
        ):
            data[col] = rng.uniform(0, 100, n)
        else:
            data[col] = rng.choice(["a", "b", "c"], n)

    # Force target columns to valid values
    data["vital_status"]      = rng.choice(["alive", "dead"], n)
    data["treatment_outcome"] = rng.choice(
        ["complete response", "partial response", "stable disease", "progressive disease"], n
    )
    data["overall_survival"]  = rng.uniform(0, 5000, n)

    df = pd.DataFrame(data)
    path = tmp_path_factory.mktemp("data") / "tcga_test.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture(scope="session")
def schema():
    """Return the canonical server schema."""
    from shared.model_schema import SERVER_SCHEMA
    return SERVER_SCHEMA


@pytest.fixture(scope="session")
def small_supernet():
    """Return a small Supernet instance (reduced dims for speed)."""
    from supernet import Supernet
    return Supernet(input_dim=32, max_depth=3, hidden_dim=16, num_toxicity_classes=4)


@pytest.fixture()
def tmp_db_path(tmp_path):
    """Return a path to a fresh temporary database.json."""
    db = {"users": [], "projects": [], "rounds_history": []}
    p = tmp_path / "database.json"
    p.write_text(json.dumps(db))
    return str(p)
