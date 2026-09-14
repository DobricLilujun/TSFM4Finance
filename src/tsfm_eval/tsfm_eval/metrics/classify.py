"""Classification metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score as _f1
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
    return float(_f1(y_true, y_pred, average=average, zero_division=0))


def precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    p, r, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return float(p), float(r)


def auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC-ROC; returns 0.5 if only one class is present."""
    try:
        if len(np.unique(y_true)) < 2:
            return 0.5
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5
