"""Task-specific metrics and the composite score."""
from __future__ import annotations

from tsfm_eval.metrics.forecast import rmse, mae, mse, mape, smape, mase
from tsfm_eval.metrics.forecast import (directional_accuracy,
                                               directional_accuracy_change,
                                               pearson_corr, rank_corr)
from tsfm_eval.metrics.classify import accuracy, f1_score, precision_recall, auc_roc
from tsfm_eval.metrics.anomaly import anomaly_f1, anomaly_precision_recall, anomaly_auc
from tsfm_eval.metrics.score import composite_score
from tsfm_eval.metrics.trading import (TRADING_COST_GRID, ic_signal, max_drawdown,
                                       trading_metrics)
from tsfm_eval.metrics.probabilistic import (QUANTILE_LEVELS, crps_from_quantiles,
                                              normalized_difference, pinball,
                                              quantile_coverage, var_es_from_quantiles, wql)

__all__ = [
    "accuracy",
    "anomaly_f1",
    "anomaly_precision_recall",
    "anomaly_auc",
    "auc_roc",
    "composite_score",
    "directional_accuracy",
    "directional_accuracy_change",
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
