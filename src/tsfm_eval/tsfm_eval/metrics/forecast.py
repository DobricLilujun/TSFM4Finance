"""Forecasting metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


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


def mase(
    pred: np.ndarray,
    truth: np.ndarray,
    history: np.ndarray | None = None,
    season: int = 1,
) -> float:
    """Mean Absolute Scaled Error."""
    if history is None or len(history) < season + 1:
        scale = float(np.mean(np.abs(np.diff(truth))))
    else:
        scale = float(np.mean(np.abs(history[season:] - history[:-season])))
    if scale == 0:
        return float(np.mean(np.abs(pred - truth)))
    return float(np.mean(np.abs(pred - truth)) / scale)


def directional_accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
    """Fraction of steps where sign(pred) == sign(truth).

    Only meaningful when the inputs are *moves* (returns, differences). For price levels it is a
    constant: every price is positive, so sign(pred) == sign(truth) == +1 and a forecast that gets
    every direction wrong still scores 1.0. Use `directional_accuracy_change` for level series.
    """
    return float(np.mean(np.sign(pred) == np.sign(truth)))


def directional_accuracy_change(pred: np.ndarray, truth: np.ndarray,
                                last_context: float | None = None) -> float:
    """Directional accuracy of the *changes*, for level series.

    The first comparison is the move from the last observed context price to the first forecast step;
    dropping it would score only the interior steps and quietly change the denominator depending on
    whether a context was passed at all.
    """
    p = np.asarray(pred, dtype=float)
    y = np.asarray(truth, dtype=float)
    if p.size == 0 or y.size == 0 or p.shape != y.shape:
        return float("nan")
    if last_context is not None and np.isfinite(float(last_context)):
        p = np.concatenate([[float(last_context)], p])
        y = np.concatenate([[float(last_context)], y])
    elif p.size < 2:
        return float("nan")          # one step of levels carries no direction at all
    dp, dy = np.diff(p), np.diff(y)
    keep = (dp != 0) & (dy != 0)
    if not keep.any():
        return float("nan")
    return float(np.mean(np.sign(dp[keep]) == np.sign(dy[keep])))


def rank_corr(pred: np.ndarray, truth: np.ndarray) -> float:
    """Spearman rank correlation; falls back to pandas rank correlation."""
    try:
        from scipy.stats import spearmanr
    except Exception:  # pragma: no cover - optional dependency
        pr = pd.Series(pred).rank()
        tt = pd.Series(truth).rank()
        return float(pr.corr(tt))
    r = spearmanr(pred, truth)[0]
    return 0.0 if r is None or np.isnan(r) else float(r)


def pearson_corr(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.corrcoef(pred, truth)[0, 1])
