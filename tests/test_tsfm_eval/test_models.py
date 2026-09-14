"""Tests for tsfm_eval model adapter and benchmark runner."""
from __future__ import annotations

import pandas as pd
import pytest

from tsfm_eval import evaluate
from tsfm_eval.data.dataset import TSDataset
from tsfm_eval.evaluation.benchmark import run_benchmark
from tsfm_eval.models.base import BaseModel, PredictionResult
from tsfm_eval.schemas import DatasetMeta, PredictOutput


class ConstantModel(BaseModel):
    def __init__(self):
        super().__init__("constant")

    def predict(self, context_df, meta, horizon=None):
        value = float(context_df[meta.target].iloc[-1])
        return PredictionResult(point=[value] * (horizon or meta.horizon))


def test_dummy_predict_evaluate():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=50),
        "close": range(50),
    })
    meta = DatasetMeta(
        name="test", domain="equity", task="forecast",
        frequency="day", n_features=1, feature_names=["close"],
        target="close", lookback=10, horizon=3,
    )
    ds = TSDataset(meta=meta, train_df=df.iloc[:30], val_df=df.iloc[30:40], test_df=df.iloc[40:])
    model = ConstantModel()
    ctx, truth = next(ds.iter_rolling("test"))
    pred = model.predict(ctx, ds.meta)
    m = evaluate(pred.to_output(), truth, ds.meta, train_df=ds.train_df)
    assert "score" in m


def test_benchmark_runner():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=80),
        "close": range(80),
    })
    meta = DatasetMeta(
        name="test", domain="equity", task="forecast",
        frequency="day", n_features=1, feature_names=["close"],
        target="close", lookback=10, horizon=3,
    )
    ds = TSDataset(meta=meta, train_df=df.iloc[:50], val_df=df.iloc[50:65], test_df=df.iloc[65:])
    results = run_benchmark(ConstantModel(), [ds])
    assert len(results) == 1
    assert results[0].model_name == "constant"
    assert 0.0 <= results[0].score <= 1.0
