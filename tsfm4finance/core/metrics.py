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
from tsfm_eval.metrics import TRADING_COST_GRID, ic_signal, max_drawdown, trading_metrics
from tsfm_eval.metrics import (QUANTILE_LEVELS, crps_from_quantiles, normalized_difference,
                                   pinball, quantile_coverage, var_es_from_quantiles, wql)

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
    "QUANTILE_LEVELS",
    "crps_from_quantiles",
    "normalized_difference",
    "pinball",
    "quantile_coverage",
    "var_es_from_quantiles",
    "wql",
    "TRADING_COST_GRID",
    "ic_signal",
    "max_drawdown",
    "trading_metrics",
]
