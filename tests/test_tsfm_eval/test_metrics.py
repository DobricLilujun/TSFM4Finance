"""Tests for tsfm_eval metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsfm_eval import evaluate
from tsfm_eval.metrics import rmse, mae, accuracy, f1_score
from tsfm_eval.schemas import DatasetMeta, PredictOutput, TaskType


def test_forecast_metrics():
    pred = PredictOutput(point=[101.0, 102.0, 103.0])
    truth = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=3), "close": [100.0, 102.0, 104.0]})
    meta = DatasetMeta(
        name="test", domain="equity", task=TaskType.FORECAST,
        frequency="day", n_features=1, feature_names=["close"],
        target="close", lookback=10, horizon=3,
    )
    m = evaluate(pred, truth, meta)
    assert "rmse" in m
    assert "score" in m
    assert 0.0 <= m["score"] <= 1.0


def test_perfect_forecast_score():
    truth = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=3), "close": [100.0, 101.0, 102.0]})
    pred = PredictOutput(point=[100.0, 101.0, 102.0])
    meta = DatasetMeta(
        name="test", domain="equity", task=TaskType.FORECAST,
        frequency="day", n_features=1, feature_names=["close"],
        target="close", lookback=10, horizon=3,
    )
    m = evaluate(pred, truth, meta)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["score"] > 0.9


def test_classification_metrics():
    truth = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=4), "label": [0, 1, 0, 1]})
    pred = PredictOutput(probabilities=[0.2, 0.8, 0.3, 0.9])
    meta = DatasetMeta(
        name="test", domain="equity", task=TaskType.CLASSIFY,
        frequency="day", n_features=1, feature_names=["label"],
        target="label", lookback=10, horizon=1,
    )
    m = evaluate(pred, truth, meta)
    assert "accuracy" in m
    assert "f1" in m
    assert 0.0 <= m["score"] <= 1.0


def test_raw_helpers():
    assert rmse(np.array([1.0]), np.array([2.0])) == pytest.approx(1.0)
    assert mae(np.array([1.0]), np.array([2.0])) == pytest.approx(1.0)
    assert accuracy(np.array([0, 1]), np.array([0, 1])) == 1.0
    assert f1_score(np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1])) == 1.0
