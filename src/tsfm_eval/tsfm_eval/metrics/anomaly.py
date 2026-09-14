"""Anomaly-detection metrics (thin wrappers over sklearn)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score as _f1
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def anomaly_f1(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
    return float(_f1(y_true, y_pred, average=average, zero_division=0))


def anomaly_precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    p, r, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return float(p), float(r)


def anomaly_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return 0.5
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5
