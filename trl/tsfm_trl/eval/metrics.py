"""Metrics shared by the evaluator and the trainers.

Forecast metrics are scale-free where possible (MASE, sMAPE, ND) so one table can
hold S&P 500 and BTC. Probabilistic metrics (pinball, CRPS, WQL) are evaluated on
the *shared* quantile grid the adapters resample onto — comparing two models on
their own grids is not a comparison.
"""
from __future__ import annotations

import numpy as np

from ..model.base import LEVELS

__all__ = ["mase", "smape", "rmse", "mae", "nd", "relative_skill", "pinball", "crps_from_quantiles",
           "wql", "quantile_coverage", "directional_accuracy", "mape", "rank_corr",
           "classification_report", "forecast_report"]


def _clean(y, p):
    """Flatten and align: a forecast that is one step short (or long) is compared on
    the overlapping tail rather than raising."""
    y = np.asarray(y, float).reshape(-1)
    p = np.asarray(p, float).reshape(-1)
    if y.shape != p.shape:
        n = min(len(y), len(p))
        y, p = y[-n:], p[-n:]
    m = np.isfinite(y) & np.isfinite(p)
    return y[m], p[m]


def rmse(y, p):
    y, p = _clean(y, p)
    return float(np.sqrt(np.mean((y - p) ** 2))) if len(y) else float("nan")


def mae(y, p):
    y, p = _clean(y, p)
    return float(np.mean(np.abs(y - p))) if len(y) else float("nan")


def mape(y, p):
    y, p = _clean(y, p)
    y = y[np.abs(y) > 1e-12]
    p = p[: len(y)]
    return float(np.mean(np.abs((y - p) / y)) * 100) if len(y) else float("nan")


def smape(y, p):
    y, p = _clean(y, p)
    d = (np.abs(y) + np.abs(p)) / 2
    m = d > 1e-12
    return float(np.mean(np.abs(y[m] - p[m]) / d[m]) * 100) if m.any() else float("nan")


def mase(y, p, train, season: int = 1):
    """Mean absolute scaled error, scaled by the train-split naive error."""
    y, p = _clean(y, p)
    tr = np.asarray(train, float)
    tr = tr[np.isfinite(tr)]
    if len(tr) < season + 1 or not len(y):
        return float("nan")
    denom = np.mean(np.abs(np.diff(tr, n=season))) if season > 1 else np.mean(np.abs(np.diff(tr)))
    if not np.isfinite(denom) or denom <= 1e-12:
        return float("nan")
    return float(np.mean(np.abs(y - p)) / denom)


def nd(y, p, train):
    """Normalised difference (relative to the naive benchmark) — the finance-bench
    favourite: < 0 means better than persistence."""
    y, p = _clean(y, p)
    y0 = np.asarray(train, float)[-1] if len(np.asarray(train, float)) else y[0]
    naive = np.mean(np.abs(y - y0))
    return float(np.mean(np.abs(y - p)) / naive - 1.0) if naive > 1e-12 else float("nan")


def relative_skill(y, p, train, season: int = 1):
    m = mase(y, p, train, season)
    return float(1.0 - m) if np.isfinite(m) else float("nan")


def directional_accuracy(y, p, anchor: float | np.ndarray | None = None):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if anchor is None:
        anchor = 0.0
    m = np.isfinite(y) & np.isfinite(p)
    if not m.any():
        return float("nan")
    return float(np.mean(np.sign(p[m] - anchor) == np.sign(y[m] - anchor)))


def rank_corr(y, p):
    from scipy.stats import spearmanr
    y, p = _clean(y, p)
    if len(y) < 3:
        return float("nan")
    r = spearmanr(y, p).statistic
    return float(r) if np.isfinite(r) else float("nan")


# ------------------------------------------------------------- probabilistic
def pinball(y, q, levels=LEVELS):
    """y (…,H), q (…,H,nq) -> mean pinball loss."""
    y = np.asarray(y, float)
    q = np.asarray(q, float)
    lv = np.asarray(levels, float)
    if q.ndim == y.ndim:
        q = q[..., None]
    if q.shape[-2] != y.shape[-1]:
        raise ValueError(f"horizon mismatch {q.shape} vs {y.shape}")
    if q.shape[-1] != lv.size:
        raise ValueError(f"quantile axis {q.shape[-1]} != levels {lv.size}")
    d = y[..., None] - q
    return float(np.nanmean(np.maximum(lv * d, (lv - 1.0) * d)))


def crps_from_quantiles(y, q, levels=LEVELS):
    """CRPS from a quantile grid (the standard quantile-based approximation used by
    GIFT-Eval-style harnesses)."""
    y = np.asarray(y, float)
    q = np.asarray(q, float)
    lv = np.asarray(levels, float)
    if q.ndim == y.ndim:
        q = q[..., None]
    d = y[..., None] - q
    pl = np.maximum(lv * d, (lv - 1.0) * d)
    return float(2.0 * np.nanmean(pl))


def wql(y, q, levels=LEVELS):
    """Weighted quantile loss with the usual 1/(2*level*(1-level))-style weights,
    which is what puts weight on the tails (VaR-relevant)."""
    y = np.asarray(y, float)
    q = np.asarray(q, float)
    lv = np.asarray(levels, float)
    d = y[..., None] - q
    pl = np.maximum(lv * d, (lv - 1.0) * d)
    w = 1.0 / (2.0 * lv * (1.0 - lv))
    w = w / w.sum()
    return float(np.nansum(pl * w))


def quantile_coverage(y, q, levels=LEVELS):
    """Empirical coverage of the central band per level: what a VaR user cares about."""
    y = np.asarray(y, float)
    q = np.asarray(q, float)
    lv = np.asarray(levels, float)
    """Coverage of the central bands that the *emitted grid* actually contains.

    The nominal value is the level pair that was found, not the band that was asked
    for: a model emitting only 0.1..0.9 cannot produce a 95 % band, and claiming
    nominal_95 for an 80 % interval is exactly the kind of interval lie a risk
    reader would act on.
    """
    out = {}
    nq = lv.size
    seen = set()
    for i in range(min(3, nq // 2)):
        j = nq - 1 - i
        if j <= i:
            break
        lo, hi = float(lv[i]), float(lv[j])
        band = int(round((hi - lo) * 100))
        if band in seen:
            continue
        seen.add(band)
        inside = (y >= q[..., i]) & (y <= q[..., j])
        out[f"coverage_{band}"] = float(np.nanmean(inside))
        out[f"nominal_{band}"] = float(hi - lo)
        out[f"levels_{band}"] = [lo, hi]
    return out


# -------------------------------------------------------------- classification
def classification_report(y_true, y_score, n_classes: int | None = None,
                          threshold: float | None = None) -> dict:
    """``y_score`` is class labels *or* a score matrix / positive-class probability.

    Pass scores whenever you have them: the derived finance labels are imbalanced,
    where ROC-AUC flatters and accuracy is meaningless, and PR-AUC — the number that
    actually discriminates a skill from a majority-class guess — can only be computed
    from scores. For >2 classes the ranking metrics are one-vs-rest weighted over the
    classes present in ``y_true``; a class with no held-out example is skipped rather
    than silently scored 0.
    """
    import numpy as np
    from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    from sklearn.preprocessing import label_binarize
    y = np.asarray(y_true).reshape(-1)
    s = np.asarray(y_score)
    k = int(n_classes or len(np.unique(y)))
    out: dict = {"pr_auc": float("nan"), "roc_auc": float("nan")}
    if s.ndim == 2:                                        # score matrix
        if k <= 2:
            prob = s[:, -1] if s.shape[1] >= 2 else s.reshape(-1)
            pred = (prob >= (threshold if threshold is not None else 0.5)).astype(int)
            y_bin = (y > 0).astype(int)
            if len(np.unique(y_bin)) == 2:
                out["pr_auc"] = float(average_precision_score(y_bin, prob))
                out["roc_auc"] = float(roc_auc_score(y_bin, prob))
        else:
            pred = s.argmax(-1)
            present = [c for c in range(k) if (y == c).any() and s.shape[1] > c]
            if len(present) >= 2:
                Yb = label_binarize(y, classes=range(k))[:, present]
                w = [int((y == c).sum()) for c in present]
                try:
                    out["pr_auc"] = float(average_precision_score(
                        Yb, s[:, present], average="weighted"))
                except ValueError:                         # a class with no positives
                    pass
                try:
                    out["roc_auc"] = float(roc_auc_score(
                        Yb, s[:, present], average="weighted"))
                except ValueError:
                    pass
    elif s.ndim == 1 and s.dtype.kind == "f" and k <= 2:   # positive-class probability
        prob = s.reshape(-1)
        pred = (prob >= (threshold if threshold is not None else 0.5)).astype(int)
        y_bin = (y > 0).astype(int)
        if len(np.unique(y_bin)) == 2:
            out["pr_auc"] = float(average_precision_score(y_bin, prob))
            out["roc_auc"] = float(roc_auc_score(y_bin, prob))
    else:                                                  # hard labels
        pred = s.reshape(-1).astype(int)
    y_bin = (y > 0).astype(int) if k <= 2 else y
    out.update({
        "accuracy": float(np.mean(pred == y)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "f1_beta2": _f_beta2(y, pred, k),
        "precision": float(precision_score(y, pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y, pred, average="macro", zero_division=0)),
        "n": int(len(y)), "n_classes": int(k),
        "base_rate": float(max(np.bincount(y, minlength=k)) / max(1, len(y))),
    })
    out.setdefault("auc_roc", out["roc_auc"])              # one spelling, kept for old rows
    return out


def _f_beta2(y, pred, k):
    """F2 (recall-weighted): the finance framing of "did we catch the moves".

    It lives in ``fbeta_score`` — ``f1_score`` does not take a ``beta`` at all, which is
    how this number silently became nan for every run once. For >2 classes it is the
    macro average of each class's one-vs-rest F2, restricted to labels that actually
    appear, so a class nobody predicted and nobody observed cannot drag the mean down.
    """
    from sklearn.metrics import fbeta_score
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        if k <= 2:
            return float(fbeta_score(y, pred, beta=2.0, pos_label=int(max(y)),
                                     zero_division=0))
        present = [int(c) for c in range(k) if (y == c).any() or (pred == c).any()]
        if len(present) < 2:
            return float("nan")
        return float(fbeta_score(y, pred, beta=2.0, average="macro", labels=present,
                                 zero_division=0))
    except Exception:                                              # noqa: BLE001
        return float("nan")


# ------------------------------------------------------------------- reports
def _mean_window(y, p, fn) -> float:
    vs = [fn(y[i], p[i]) for i in range(len(y))]
    vs = [float(v) for v in vs if v is not None and np.isfinite(v)]
    return float(np.mean(vs)) if vs else float("nan")


def _dir_acc(y, p, anchor=None) -> float:
    if anchor is None:
        return _mean_window(np.diff(y, axis=1), np.diff(p, axis=1), directional_accuracy)
    a = np.asarray(anchor, float).reshape(len(y))
    vs = [directional_accuracy(y[i], p[i], anchor=float(a[i])) for i in range(len(y))]
    vs = [float(v) for v in vs if np.isfinite(v)]
    return float(np.mean(vs)) if vs else float("nan")


def forecast_report(y, pred, train, *, season: int = 1, anchor=None) -> dict:
    """The point-forecast block of an evaluation row, plus the probabilistic block when
    the model produced quantiles.

    ``dir_acc`` and ``rank_corr`` are per-window means: pooling them across windows at
    different price levels would measure the level differences, not the forecast.
    With ``anchor`` (usually each window's last observed value) dir_acc is "did the
    forecast point the right way away from where the series stood"; without one it
    compares step-to-step directions inside the horizon, which needs no context.
    """
    y = np.asarray(y, float)
    p = np.asarray(pred.point if hasattr(pred, "point") else pred, float)
    if y.ndim == 1:
        y, p = y[None, :], p[None, :]
    q = getattr(pred, "quantiles", None) if hasattr(pred, "quantiles") else None
    out = {"rmse": rmse(y, p), "mae": mae(y, p), "mase": mase(y, p, train, season),
           "smape": smape(y, p), "nd": nd(y, p, train),
           "relative_skill": relative_skill(y, p, train, season),
           "dir_acc": _dir_acc(y, p, anchor),
           "rank_corr": _mean_window(y, p, rank_corr)}
    if q is not None:
        lv = getattr(pred, "levels", LEVELS)
        out["pinball"] = pinball(y, q, lv)
        out["crps"] = crps_from_quantiles(y, q, lv)
        out["wql"] = wql(y, q, lv)
        out.update(quantile_coverage(y, q, lv))
    return out


def classification_metrics(**kw) -> dict:
    return classification_report(**kw)
