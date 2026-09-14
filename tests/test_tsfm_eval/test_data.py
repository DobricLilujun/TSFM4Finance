"""Tests for tsfm_eval data interface."""
from __future__ import annotations

import pytest

import tsfm_eval
from tsfm_eval import TSDataset, load_dataset, load_meta, list_datasets


def test_list_datasets_real():
    names = [m.name for m in list_datasets(open_only=True)]
    assert "sp500_daily" in names


def test_load_meta_real():
    meta = load_meta("sp500_daily")
    assert meta.task.value == "forecast"
    assert meta.target == "close"


def test_load_dataset_real():
    ds = load_dataset("sp500_daily")
    assert ds.name == "sp500_daily"
    assert len(ds.train_df) > 0
    assert len(ds.test_df) > 0


def test_dataset_splits(synthetic_dataset):
    meta, data_root = synthetic_dataset
    ds = TSDataset(name="test_daily", data_root=data_root)
    assert len(ds.train_df) == 80
    assert len(ds.val_df) == 15
    assert len(ds.test_df) == 55


def test_rolling_windows(synthetic_dataset):
    meta, data_root = synthetic_dataset
    ds = TSDataset(name="test_daily", data_root=data_root)
    windows = list(ds.iter_rolling("test"))
    # The first lookback rows are consumed as history, so the number of
    # complete (context, truth) windows is n_test - lookback - horizon + 1.
    expected = len(ds.test_df) - ds.lookback - ds.horizon + 1
    assert len(windows) == expected
    ctx, truth = windows[0]
    assert len(ctx) == ds.lookback
    assert len(truth) == ds.horizon


def test_registry_open_only(synthetic_dataset):
    meta, data_root = synthetic_dataset
    names = [m.name for m in tsfm_eval.list_datasets(open_only=True, data_root=data_root)]
    assert "test_daily" in names
