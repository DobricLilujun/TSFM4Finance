"""Evaluation metrics for Finance TSFMs Arena.

Implements the spec from docs/02_unified_interface_spec.md §2, §5.
All metrics are pure functions over numpy arrays so they are testable and fast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from .schemas import DatasetMeta, PredictOutput, TaskType


# --------------------------------------------------------------------------- #
# Raw numeric metrics
# --------------------------------------------------------------------------- #
def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def mae(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - truth)))


def mse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean((pred - truth) ** 2))


def mape(pred: np.ndarray, truth: np.ndarray) -> float:
    mask = truth != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((truth[mask] - pred[mask]) / truth[mask])) * 100)


def smape(pred: np.ndarray, truth: np.ndarray) -> float:
    denom = np.abs(pred) + np.abs(truth)
    mask = denom != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((truth[mask] - pred[mask]) / (denom[mask] / 2))) * 100)


def mase(pred: np.ndarray, truth: np.ndarray, history: np.ndarray | None = None,
         season: int = 1) -> float:
    """Mean Absolute Scaled Error. Requires a naive in-sample scale if history
    given, else falls back to a simple scale = mean abs diff of truth."""
    if history is None or len(history) < season + 1:
        scale = float(np.mean(np.abs(np.diff(truth))))
    else:
        scale = float(np.mean(np.abs(history[season:] - history[:-season])))
    if scale == 0:
        return float(np.mean(np.abs(pred - truth)))
    return float(np.mean(np.abs(pred - truth)) / scale)


def directional_accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
    """Directional accuracy: fraction where sign(pred) matches sign(truth).
    Used for return/direction prediction."""
    pd_sign = np.sign(pred)
    t_sign = np.sign(truth)
    return float(np.mean(pd_sign == t_sign))


def rank_corr(pred: np.ndarray, truth: np.ndarray) -> float:
    """Spearman rank correlation between predicted and true series."""
    try:
        from scipy.stats import spearmanr
    except Exception:
        # fallback: Pearson on ranks via pandas
        pr = pd.Series(pred).rank()
        tt = pd.Series(truth).rank()
        return float(pr.corr(tt))
    r = spearmanr(pred, truth)[0]
    return 0.0 if r is None or np.isnan(r) else float(r)


def pearson_corr(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.corrcoef(pred, truth)[0, 1])


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, average="macro") -> float:
    from sklearn.metrics import f1_score as _f1
    return float(_f1(y_true, y_pred, average=average, zero_division=0))


def precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import precision_recall_fscore_support
    p, r, _, _ = precision_recall_fscore_support(y_true, y_pred, average="macro",
                                                 zero_division=0)
    return float(p), float(r)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        if len(np.unique(y_true)) < 2:
            return 0.5
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5


# --------------------------------------------------------------------------- #
# Composite score (0..1, higher is better) — spec §5
# --------------------------------------------------------------------------- #
def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _score_forecast(m: dict[str, float]) -> float:
    s = 0.35 * (1 - _clip01(m.get("mase", 1.0)))
    s += 0.25 * m.get("directional_accuracy", 0.5)
    s += 0.20 * max(0.0, m.get("return_rank_corr", 0.0))
    s += 0.20 * (1 - _clip01(m.get("smape", 100.0) / 100.0))
    return round(_clip01(s), 4)


def _score_classify(m: dict[str, float]) -> float:
    return round(_clip01(0.6 * m.get("f1", 0.0) + 0.4 * m.get("auc", 0.5)), 4)


def _score_anomaly(m: dict[str, float]) -> float:
    return round(_clip01(0.6 * m.get("f1", 0.0) + 0.4 * m.get("auc", 0.5)), 4)


_COMPOSITE = {
    TaskType.FORECAST: _score_forecast,
    TaskType.CLASSIFY: _score_classify,
    TaskType.ANOMALY: _score_anomaly,
}


def composite_score(task: TaskType, m: dict[str, float]) -> float:
    return _COMPOSITE[task](m)


# --------------------------------------------------------------------------- #
# Public evaluate()
# --------------------------------------------------------------------------- #
def evaluate(
    pred: PredictOutput,
    truth_df: pd.DataFrame,
    meta: DatasetMeta,
    train_df: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Evaluate a PredictOutput against ground truth for a dataset.

    Returns a dict of metrics plus a composite 'score'.
    """
    m: dict[str, float] = {}

    # ---- forecast ----
    if meta.task == TaskType.FORECAST:
        truth = _extract_forecast_truth(truth_df, meta)
        pred_pts = np.asarray(_flatten(pred.point or []), dtype=float)
        # align length (pad/truncate)
        n = min(len(truth), len(pred_pts))
        if n == 0:
            return {"score": 0.0}
        truth = truth[:n]
        pred_pts = pred_pts[:n]

        m["rmse"] = rmse(pred_pts, truth)
        m["mae"] = mae(pred_pts, truth)
        m["mse"] = mse(pred_pts, truth)
        m["mape"] = mape(pred_pts, truth)
        m["smape"] = smape(pred_pts, truth)

        # MASE scale from train history if available
        hist = None
        if train_df is not None:
            hist = _extract_forecast_truth(train_df, meta)
        m["mase"] = mase(pred_pts, truth, history=hist)

        # financial extras
        m["directional_accuracy"] = directional_accuracy(pred_pts, truth)
        m["return_rank_corr"] = rank_corr(pred_pts, truth)
        m["pearson"] = pearson_corr(pred_pts, truth)
        m["score"] = composite_score(TaskType.FORECAST, m)
        return m

    # ---- classify ----
    if meta.task == TaskType.CLASSIFY:
        truth = _extract_labels(truth_df, meta)
        if pred.probabilities is not None:
            y_score = np.asarray(pred.probabilities, dtype=float)
        else:
            y_score = np.asarray(pred.point or [0.5], dtype=float)
        y_pred = np.argmax(y_score) if y_score.ndim == 1 else y_score
        # If scalar probabilities (n_classes,), y_pred is an int; truth is array
        if np.ndim(y_pred) == 0:
            y_pred_arr = np.array([y_pred])
        else:
            y_pred_arr = np.asarray(y_pred).flatten()
        t = truth[:len(y_pred_arr)]
        p = y_pred_arr[:len(t)]
        m["accuracy"] = accuracy(t, p)
        m["f1"] = f1_score(t, p)
        p_, r_ = precision_recall(t, p)
        m["precision"] = p_
        m["recall"] = r_
        m["auc"] = auc_roc(t, y_score.flatten() if y_score.ndim > 1 else _prob_to_score(y_score))
        m["score"] = composite_score(TaskType.CLASSIFY, m)
        return m

    # ---- anomaly ----
    if meta.task == TaskType.ANOMALY:
        truth = _extract_labels(truth_df, meta)
        y_pred = np.asarray(pred.point or [0.0], dtype=float)
        y_pred_bin = (y_pred > 0.5).astype(int) if (pred.point and pred.point[0] <= 1) else y_pred.astype(int)
        t = truth[:len(y_pred_bin)]
        p = y_pred_bin[:len(t)]
        m["f1"] = f1_score(t, p)
        p_, r_ = precision_recall(t, p)
        m["precision"] = p_
        m["recall"] = r_
        m["auc"] = auc_roc(t, y_pred)
        m["score"] = composite_score(TaskType.ANOMALY, m)
        return m

    return m


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _flatten(x) -> list[float]:
    out: list[float] = []
    for v in x:
        if isinstance(v, (list, tuple, np.ndarray)):
            out.extend(float(z) for z in v)
        else:
            out.append(float(v))
    return out


def _extract_forecast_truth(df: pd.DataFrame, meta: DatasetMeta) -> np.ndarray:
    col = meta.target
    if col in df.columns:
        return df[col].to_numpy(dtype=float)
    # fallback: last numeric column
    return df.select_dtypes(include="number").iloc[:, -1].to_numpy(dtype=float)


def _extract_labels(df: pd.DataFrame, meta: DatasetMeta) -> np.ndarray:
    for c in ("label", "anomaly"):
        if c in df.columns:
            return df[c].to_numpy()
    return np.zeros(len(df))


def _prob_to_score(y_score: np.ndarray) -> np.ndarray:
    """Convert class-probability vector to positive-class score for AUC."""
    return y_score