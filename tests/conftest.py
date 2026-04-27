"""
conftest.py — shared pytest fixtures for Heart Disease dashboard tests.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# ── Reference training-like dataframe ─────────────────────────────────────
NUMERIC_COLS  = ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]
CAT_COLS_BASE = [
    "Sex", "Chest pain type", "FBS over 120", "EKG results",
    "Exercise angina", "Slope of ST", "Number of vessels fluro", "Thallium",
]
ALL_COLS = NUMERIC_COLS + CAT_COLS_BASE


@pytest.fixture()
def ref_df():
    """Synthetic reference distribution (300 rows)."""
    rng = np.random.default_rng(42)
    data = {
        "Age":                       rng.integers(30, 80, 300),
        "BP":                        rng.integers(90, 180, 300),
        "Cholesterol":               rng.integers(150, 400, 300),
        "Max HR":                    rng.integers(70, 200, 300),
        "ST depression":             rng.uniform(0, 5, 300).round(1),
        "Sex":                       rng.choice([0, 1], 300),
        "Chest pain type":           rng.choice([1, 2, 3, 4], 300),
        "FBS over 120":              rng.choice([0, 1], 300),
        "EKG results":               rng.choice([0, 1, 2], 300),
        "Exercise angina":           rng.choice([0, 1], 300),
        "Slope of ST":               rng.choice([1, 2, 3], 300),
        "Number of vessels fluro":   rng.choice([0, 1, 2, 3], 300),
        "Thallium":                  rng.choice([3, 6, 7], 300),
    }
    return pd.DataFrame(data)


@pytest.fixture()
def batch_same(ref_df):
    """Batch sampled from same distribution → no drift expected."""
    return ref_df.sample(60, random_state=99).reset_index(drop=True)


@pytest.fixture()
def batch_shifted():
    """Batch with numeric columns shifted significantly → drift expected."""
    rng = np.random.default_rng(7)
    data = {
        "Age":                       rng.integers(65, 90, 60),   # shifted up
        "BP":                        rng.integers(160, 200, 60), # shifted up
        "Cholesterol":               rng.integers(350, 500, 60), # shifted up
        "Max HR":                    rng.integers(50, 80, 60),   # shifted down
        "ST depression":             rng.uniform(3, 6, 60).round(1),
        "Sex":                       rng.choice([0, 1], 60),
        "Chest pain type":           rng.choice([1, 2, 3, 4], 60),
        "FBS over 120":              rng.choice([0, 1], 60),
        "EKG results":               rng.choice([0, 1, 2], 60),
        "Exercise angina":           rng.choice([0, 1], 60),
        "Slope of ST":               rng.choice([1, 2, 3], 60),
        "Number of vessels fluro":   rng.choice([0, 1, 2, 3], 60),
        "Thallium":                  rng.choice([3, 6, 7], 60),
    }
    return pd.DataFrame(data)


@pytest.fixture()
def batch_cat_shifted(ref_df):
    """Batch where categoricals are dominated by a single class → Chi² drift."""
    df = ref_df.sample(60, random_state=5).reset_index(drop=True).copy()
    df["Chest pain type"] = 4       # all asymptomatic — strong imbalance
    df["Thallium"]        = 7       # all reversible defect
    return df


@pytest.fixture()
def metrics_dict():
    return {
        "overall_oof_auc": 0.912345,
        "overall_oof_ap":  0.887654,
        "mean_fold_auc":   0.908000,
        "std_fold_auc":    0.012000,
        "fold_scores":     [0.90, 0.91, 0.92, 0.89, 0.93],
        "mlflow_run_id":   "abc123",
    }


@pytest.fixture()
def metrics_file(tmp_path, metrics_dict):
    """Write metrics.json to a tmp dir and return its path."""
    p = tmp_path / "metrics.json"
    p.write_text(json.dumps(metrics_dict))
    return p


@pytest.fixture()
def mock_mlflow_response():
    """Mock requests.post returning a valid MLflow /invocations response."""
    mock = MagicMock()
    mock.status_code = 200
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {"predictions": [0.73]}
    return mock
