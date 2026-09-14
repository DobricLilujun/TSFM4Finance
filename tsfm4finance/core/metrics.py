"""Backward-compatible metrics module.

The canonical implementation now lives in ``tsfm_eval.metrics``.  This module
re-exports the unified ``evaluate()`` dispatcher and keeps the raw metric
helpers available under the old names.
"""
from __future__ import annotations

from tsfm_eval.metrics import (
    accuracy,
    anomaly_auc,
    anomaly_f1,
    anomaly_precision_recall,
    auc_roc,
    composite_score,
    directional_accuracy,
    f1_score,
    mae,
    mape,
    mase,
    mse,
    pearson_corr,
    precision_recall,
    rank_corr,
    rmse,
    smape,
)
from tsfm_eval.evaluation.evaluator import evaluate

__all__ = [
    "accuracy",
    "anomaly_auc",
    "anomaly_f1",
    "anomaly_precision_recall",
    "auc_roc",
    "composite_score",
    "directional_accuracy",
    "evaluate",
    "f1_score",
    "mae",
    "mape",
    "mase",
    "mse",
    "pearson_corr",
    "precision_recall",
    "rank_corr",
    "rmse",
    "smape",
]
