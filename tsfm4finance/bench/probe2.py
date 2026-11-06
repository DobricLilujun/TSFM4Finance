"""Discriminative-task helpers shared by the sweep worker (spec §2.2).

Why a separate module: the classification/anomaly routes must be *identical*
across models (same features, same CV, same thresholds), otherwise the model
comparison is contaminated by the classifier.  Everything here is out-of-fold:
predictions are produced by ``cross_val_predict`` on the evaluation windows, so
no window is ever scored by a model fitted on that same window.
"""
from __future__ import annotations

import numpy as np

MIN_TRAIN = 20


# ---------------------------------------------------------------------- features
def stat_features(ctx: np.ndarray) -> np.ndarray:
    """Look-back-only features: the honest statistical baseline feature block."""
    x = np.asarray(ctx, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.zeros(20)
    lr = np.diff(np.log(np.maximum(x, 1e-12)))
    f = []

    def _mom(k):
        return float(np.log(max(x[-1], 1e-12) / max(x[-1 - k], 1e-12))) if len(x) > k else 0.0

    def _vol(k):
        return float(np.std(lr[-k:], ddof=1)) if len(lr) >= 2 else 0.0

    f += [lr[-1], _mom(1)]
    for k in (5, 21, 63):
        f += [_mom(k), _vol(k)]
    f += [float(np.mean(lr)), float(np.std(lr, ddof=1)) if len(lr) > 1 else 0.0]
    f += [float(np.mean(lr[-21:])), float(np.std(lr[-21:], ddof=1)) if len(lr) >= 3 else 0.0]
    m20 = float(np.mean(x[-20:])) if len(x) >= 20 else float(x[-1])
    m60 = float(np.mean(x[-60:])) if len(x) >= 60 else float(x[-1])
    f += [float(np.log(max(x[-1], 1e-12) / max(m20, 1e-12))),
          float(np.log(max(x[-1], 1e-12) / max(m60, 1e-12)))]
    s = float(np.std(lr, ddof=1)) if len(lr) > 1 else 1.0
    f += [float(lr[-1] / s) if s > 0 else 0.0]
    f += [float(np.mean(lr**3) / s**3) if s > 0 else 0.0,
          float(np.mean(lr**4) / s**4) if s > 0 else 0.0]
    up = float(np.mean([1.0 if v > 0 else 0.0 for v in lr[-20:]])) if len(lr) >= 2 else 0.5
    f += [up]
    rng = float((np.max(x[-63:]) - np.min(x[-63:])) / max(x[-1], 1e-12)) if len(x) >= 2 else 0.0
    f += [rng]
    f += [float(np.median(lr))]
    f += [float(np.percentile(np.abs(lr), 90))]
    f += [float(np.corrcoef(lr[:-1], lr[1:])[0, 1]) if len(lr) > 3 else 0.0]
    f += [float(len(x))]
    return np.nan_to_num(np.array(f, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)[:20]


def zscore_scores(contexts) -> np.ndarray:
    """Model-free anomaly score: |last return| in trailing-sigma units."""
    out = []
    for c in contexts:
        x = np.asarray(c, dtype=float)
        x = x[np.isfinite(x)]
        if len(x) < 3:
            out.append(0.0)
            continue
        lr = np.diff(np.log(np.maximum(x, 1e-12)))
        sd = float(np.std(lr, ddof=1)) or 1e-9
        out.append(abs(float(lr[-1])) / sd)
    return np.array(out)


# ----------------------------------------------------------------------- fitting
def _safe_fit(clf, X, y):
    """Guard the two failure modes that would otherwise kill a whole cell:
    one-class folds and degenerate feature matrices."""
    from sklearn.base import clone
    y = np.asarray(y)
    if len(np.unique(y)) < 2 or len(y) < MIN_TRAIN:
        return None
    try:
        return clone(clf).fit(X, y)
    except Exception:                                              # noqa: BLE001
        return None


def _cv_predict(clf, X, y, n_splits: int = 5):
    """Out-of-fold probabilities (the anti-contamination rule for probes)."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    y = np.asarray(y)
    n_splits = int(min(n_splits, max(2, min(np.bincount(y.astype(int)).max(), len(y)))))
    if n_splits < 2 or len(np.unique(y)) < 2:
        return None
    try:
        return cross_val_predict(clone(clf), X, y, cv=StratifiedKFold(n_splits=n_splits),
                                 method="predict_proba", n_jobs=1)
    except Exception:                                              # noqa: BLE001
        return None


def fit_eval(feats, wins, proba, labels, *, route: str = "") -> dict:
    """Score one classification cell.  ``proba`` is supplied by native heads."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(labels, dtype=int)
    classes = np.unique(y)
    out: dict = {"route": route, "n": int(len(y)), "n_classes": int(len(classes)),
                 "class_counts": {int(c): int((y == c).sum()) for c in classes}}
    if proba is not None:
        P = np.asarray(proba, dtype=float)
        pred = P.argmax(axis=1)
        out["classifier"] = "native_head"
    else:
        X = np.asarray(feats, dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        best = None
        for name, clf in (("logit", make_pipeline(StandardScaler(),
                                                  LogisticRegression(max_iter=2000, C=0.5))),
                          ("gbm", HistGradientBoostingClassifier(max_iter=200, max_depth=4,
                                                                 learning_rate=0.05))):
            P = _cv_predict(clf, X, y)
            if P is None:
                continue
            f1 = _f1(y, P.argmax(axis=1), len(classes))
            if best is None or f1 > best[2]:
                best = (name, P, f1)
        if best is None:
            out["error"] = "single-class-or-too-few-samples"
            return out
        out["classifier"], P = best[0], best[1]
        pred = P.argmax(axis=1)

    from .finance_metrics import pr_auc
    out["accuracy"] = float(np.mean(pred == y))
    out["balanced_accuracy"] = _balanced_acc(y, pred, len(classes))
    out["f1_macro"] = _f1(y, pred, len(classes))
    if len(classes) == 2:
        out["f1"] = float(_f1_binary(y, pred))
        out["auc_roc"] = _auc(y, P[:, 1] if P.ndim == 2 and P.shape[1] == 2 else pred.astype(float))
        out["pr_auc"] = pr_auc(y, P[:, 1] if P.ndim == 2 and P.shape[1] == 2
                               else pred.astype(float))
        # most models are evaluated on an imbalanced label: report both
        out["majority_acc"] = float(max(np.mean(y == 0), np.mean(y == 1)))
    else:
        from sklearn.metrics import f1_score
        out["f1_macro"] = float(f1_score(y, pred, average="macro"))
    return out


def _f1(y, pred, k: int) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y, pred, average="macro", labels=list(range(k))))


def _f1_binary(y, pred) -> float:
    tp = float(np.sum((pred == 1) & (y == 1)))
    fp = float(np.sum((pred == 1) & (y == 0)))
    fn = float(np.sum((pred == 0) & (y == 1)))
    if tp == 0 and (fp > 0 or fn > 0):
        return 0.0
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def _auc(y, score) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y, score))
    except Exception:                                              # noqa: BLE001
        return float("nan")


def _balanced_acc(y, pred, k: int) -> float:
    accs = [float(np.mean(pred[y == c] == c)) for c in range(k) if np.any(y == c)]
    return float(np.mean(accs)) if accs else float("nan")


# ------------------------------------------------------------------- anomaly
def anomaly_eval(scores, labels) -> dict:
    """Rank-based anomaly metrics: AUC/PR-AUC are threshold-free, F1 is reported
    at the F1-optimal cut and at a fixed 5% flag rate (a desk-realistic budget)."""
    from .finance_metrics import fbeta, pr_auc
    from sklearn.metrics import average_precision_score, roc_auc_score

    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    keep = np.isfinite(s) & np.isfinite(y)
    s, y = s[keep], y[keep]
    out: dict = {"n": int(len(y)), "positives": int(y.sum()),
                 "base_rate": float(np.mean(y)) if len(y) else float("nan")}
    if len(y) == 0 or np.unique(y).size < 2:
        out["error"] = "single-class-labels"
        return out
    out["auc_roc"] = float(roc_auc_score(y, s))
    out["pr_auc"] = float(average_precision_score(y, s)) if hasattr(
        average_precision_score, "__call__") else pr_auc(y, s)
    best = (0.0, None)
    for thr in np.unique(np.r_[np.quantile(s, np.linspace(0.5, 0.999, 60)), s.max()]):
        pred = (s >= thr).astype(int)
        f1 = _f1_binary(y, pred)
        if f1 > best[0]:
            best = (f1, float(thr))
    out["f1_best"] = float(best[0])
    out["f1_threshold"] = best[1]
    thr95 = float(np.quantile(s, 0.95))
    out["f1_at_5pct_flag"] = float(_f1_binary(y, (s >= thr95).astype(int)))
    out["fbeta2_at_5pct"] = float(fbeta(y, (s >= thr95).astype(int), beta=2.0))
    return out
