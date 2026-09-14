"""Task-specific metrics and the composite score."""
from __future__ import annotations

from tsfm_eval.metrics.forecast import rmse, mae, mse, mape, smape, mase
from tsfm_eval.metrics.forecast import directional_accuracy, rank_corr, pearson_corr
from tsfm_eval.metrics.classify import accuracy, f1_score, precision_recall, auc_roc
from tsfm_eval.metrics.anomaly import anomaly_f1, anomaly_precision_recall, anomaly_auc
from tsfm_eval.metrics.score import composite_score

__all__ = [
    "accuracy",
    "anomaly_f1",
    "anomaly_precision_recall",
    "anomaly_auc",
    "auc_roc",
    "composite_score",
    "directional_accuracy",
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
