"""Unified evaluate() dispatcher."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tsfm_eval.schemas import DatasetMeta, PredictOutput, TaskType
from tsfm_eval.metrics.forecast import (
    directional_accuracy,
    mae,
    mape,
    mase,
    mse,
    pearson_corr,
    rank_corr,
    rmse,
    smape,
)
from tsfm_eval.metrics.classify import accuracy, auc_roc, f1_score, precision_recall
from tsfm_eval.metrics.anomaly import anomaly_auc, anomaly_f1, anomaly_precision_recall
from tsfm_eval.metrics.score import composite_score


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
    return df.select_dtypes(include="number").iloc[:, -1].to_numpy(dtype=float)


def _extract_labels(df: pd.DataFrame, meta: DatasetMeta) -> np.ndarray:
    for c in ("label", "anomaly"):
        if c in df.columns:
            return df[c].to_numpy()
    return np.zeros(len(df))


def _prob_to_score(y_score: np.ndarray) -> np.ndarray:
    return y_score


def evaluate(
    pred: PredictOutput,
    truth_df: pd.DataFrame,
    meta: DatasetMeta,
    train_df: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Evaluate a PredictOutput against ground truth for a dataset.

    Returns a dict of metrics plus a composite ``score``.
    """
    m: dict[str, float] = {}

    if meta.task == TaskType.FORECAST:
        truth = _extract_forecast_truth(truth_df, meta)
        pred_pts = np.asarray(_flatten(pred.point or []), dtype=float)
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

        hist = _extract_forecast_truth(train_df, meta) if train_df is not None else None
        m["mase"] = mase(pred_pts, truth, history=hist)

        m["directional_accuracy"] = directional_accuracy(pred_pts, truth)
        m["return_rank_corr"] = rank_corr(pred_pts, truth)
        m["pearson"] = pearson_corr(pred_pts, truth)
        m["score"] = composite_score(TaskType.FORECAST, m)
        return m

    if meta.task == TaskType.CLASSIFY:
        truth = _extract_labels(truth_df, meta)
        if pred.probabilities is not None:
            y_score = np.asarray(pred.probabilities, dtype=float)
        else:
            y_score = np.asarray(pred.point or [0.5], dtype=float)
        y_pred = np.argmax(y_score) if y_score.ndim == 1 else y_score
        if np.ndim(y_pred) == 0:
            y_pred_arr = np.array([y_pred])
        else:
            y_pred_arr = np.asarray(y_pred).flatten()
        t = truth[: len(y_pred_arr)]
        p = y_pred_arr[: len(t)]
        m["accuracy"] = accuracy(t, p)
        m["f1"] = f1_score(t, p)
        p_, r_ = precision_recall(t, p)
        m["precision"] = p_
        m["recall"] = r_
        score_flat = y_score.flatten() if y_score.ndim > 1 else _prob_to_score(y_score)
        m["auc"] = auc_roc(t, score_flat)
        m["score"] = composite_score(TaskType.CLASSIFY, m)
        return m

    if meta.task == TaskType.ANOMALY:
        truth = _extract_labels(truth_df, meta)
        y_pred = np.asarray(pred.point or [0.0], dtype=float)
        y_pred_bin = (y_pred > 0.5).astype(int) if (pred.point and pred.point[0] <= 1) else y_pred.astype(int)
        t = truth[: len(y_pred_bin)]
        p = y_pred_bin[: len(t)]
        m["f1"] = anomaly_f1(t, p)
        p_, r_ = anomaly_precision_recall(t, p)
        m["precision"] = p_
        m["recall"] = r_
        m["auc"] = anomaly_auc(t, y_pred)
        m["score"] = composite_score(TaskType.ANOMALY, m)
        return m

    return m
