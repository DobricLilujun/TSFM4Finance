"""Non-forecasting task heads on top of TimesFM (classify / anomaly) + statistical baselines.

Two ways to use a forecasting TSFM for a discriminative task (both are evaluated, so the
report can say which one actually pays):

  A. forecast-feature head: run the TSFM at each origin, turn its point forecast and its
     0.1/0.9 predictive band into features (predicted return, predicted vol, band width,
     signal sign, ...), then fit a small supervised classifier on TRAIN-split origins.
  B. embedding linear probe: mean-pool the decoder `output_embeddings` of the last patches
     and fit a linear classifier on TRAIN-split origins.

Baselines: the standard statistical feature set (lagged returns + rolling vol + momentum)
with logistic regression and gradient boosting, plus a majority-class reference.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SKILL_FEATURES = [1, 5, 10, 21, 63]


def stat_features(close: np.ndarray) -> np.ndarray:
    """Rows of [log returns at lags, rolling vol(5,21), 21d momentum] for each t.

    close: history up to and including t. Only past information is used.
    """
    c = np.asarray(close, dtype=float)
    if len(c) < 64:
        c = np.r_[np.full(64 - len(c), c[0] if len(c) else 0.0), c]
    lr = np.r_[0.0, np.diff(np.log(np.maximum(c, 1e-12)))]
    feats = []
    for k in SKILL_FEATURES:
        feats.append(lr[-k:].sum() if k <= len(lr) else 0.0)
    v5 = float(np.std(lr[-5:], ddof=1)) if len(lr) >= 5 else 0.0
    v21 = float(np.std(lr[-21:], ddof=1)) if len(lr) >= 21 else 0.0
    v63 = float(np.std(lr[-63:], ddof=1)) if len(lr) >= 63 else 0.0
    mom = (float(np.log(max(c[-1], 1e-12)) - np.log(max(c[-21], 1e-12)))) if len(c) >= 21 else 0.0
    feats += [v5, v21, v63, mom]
    return np.array(feats, dtype=float)


def forecast_features(context: np.ndarray, point: np.ndarray, quantiles: np.ndarray | None,
                      h: int) -> np.ndarray:
    """Turn one TSFM forecast into classifier features (all in log-return space)."""
    p0 = float(context[-1])
    pt = np.asarray(point, dtype=float)
    pred_ret = float(np.log(max(pt[-1], 1e-9) / max(p0, 1e-9)))
    pred_ret_path = float(np.log(max(pt[0], 1e-9) / max(p0, 1e-9)))
    width = 0.0
    if quantiles is not None:
        Q = np.asarray(quantiles, dtype=float)          # (h, 9)
        lo = Q[:, 0].max() if Q.ndim == 2 else 0.0
        hi = Q[:, -1].max() if Q.ndim == 2 else 0.0
        width = float((hi - lo) / max(p0, 1e-9))
    lr = np.r_[0.0, np.diff(np.log(np.maximum(context, 1e-12)))]
    vol21 = float(np.std(lr[-21:], ddof=1)) if len(lr) >= 21 else 0.0
    mom5 = float(lr[-5:].sum()) if len(lr) >= 5 else 0.0
    mom21 = float(lr[-21:].sum()) if len(lr) >= 21 else 0.0
    return np.array([pred_ret, pred_ret_path, np.sign(pred_ret), width,
                     width / max(vol21 * np.sqrt(h), 1e-9), vol21, mom5, mom21], dtype=float)


def make_clf(kind: str = "logistic", seed: int = 0):
    if kind == "logistic":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, C=1.0, random_state=seed))
    if kind == "gbm":
        return make_pipeline(HistGradientBoostingClassifier(max_depth=4, max_iter=200,
                                                            learning_rate=0.05,
                                                            random_state=seed))
    raise ValueError(f"unknown classifier {kind!r}")


def eval_classifier(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None,
                    n_classes: int = 2) -> dict:
    from tsfm4finance.core.metrics import accuracy, f1_score, precision_recall, auc_roc
    out: dict = {
        "accuracy": accuracy(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro"),
    }
    p, r = precision_recall(y_true, y_pred)
    out["precision"], out["recall"] = p, r
    if n_classes == 2:
        maj = float(np.mean(y_true == np.bincount(np.asarray(y_true, dtype=int)).argmax()))
        out["majority_accuracy"] = maj
        if y_score is not None:
            out["auc"] = auc_roc(y_true, y_score)
    else:
        out["majority_accuracy"] = float(np.mean(
            y_true == np.bincount(np.asarray(y_true, dtype=int)).argmax()))
        if y_score is not None:
            from sklearn.metrics import roc_auc_score
            try:
                out["auc"] = float(roc_auc_score(
                    np.eye(n_classes)[np.asarray(y_true, dtype=int)], y_score))
            except Exception:
                out["auc"] = float("nan")
    return out


def safe_fit(kind: str, X: np.ndarray, y: np.ndarray):
    """Fit ``make_clf(kind)`` unless the fit labels contain a single class
    (e.g. a training split with no extreme events), in which case return None."""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return None
    clf = make_clf(kind)
    clf.fit(np.asarray(X), y)
    return clf
