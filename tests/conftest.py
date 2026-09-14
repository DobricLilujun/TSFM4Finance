"""Shared pytest fixtures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Create a small open forecast dataset on disk and return its metadata."""
    data_root = tmp_path
    d = data_root / "open" / "test_daily"
    d.mkdir(parents=True)
    n = 150
    meta = {
        "name": "test_daily",
        "open": True,
        "domain": "equity",
        "task": "forecast",
        "frequency": "day",
        "target": "close",
        "horizon": 3,
        "lookback": 20,
        "rolling": True,
        "n_features": 1,
        "feature_names": ["close"],
        "n_obs": n,
        "license": "test",
    }
    (d / "meta.json").write_text(json.dumps(meta))
    t = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=n, freq="D"),
        "close": np.linspace(100, 160, n) + 5 * np.sin(np.linspace(0, 20, n)),
    })
    train = t.iloc[:80]
    val = t.iloc[80:95]
    test = t.iloc[95:]
    for name, df in (("train", train), ("validation", val), ("test", test)):
        df.to_parquet(d / f"{name}.parquet")
    return meta, data_root
