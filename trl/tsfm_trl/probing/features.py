"""Feature blocks for the *probe* classification route (frozen model + shallow
classifier) and the honest statistical baseline it must beat.

Identical features across models on purpose: if the classifier differs per model,
the comparison measures the classifier, not the model.
"""
from __future__ import annotations

import numpy as np

MIN_TRAIN = 20


def stat_features(ctx: np.ndarray) -> np.ndarray:
    """Look-back-only features: momentum, multi-horizon volatility, position vs
    moving averages, standardised return, skew, kurtosis, up-share. No future
    information by construction."""
    x = np.asarray(ctx, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.zeros(21)
    lr = np.diff(np.log(np.maximum(x, 1e-12)))

    def _mom(k):
        return float(np.log(max(x[-1], 1e-12) / max(x[-1 - k], 1e-12))) if len(x) > k else 0.0

    def _vol(k):
        return float(np.std(lr[-k:], ddof=1)) if len(lr) >= 2 else 0.0

    f = [lr[-1], _mom(1)]
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
    f += [float(np.mean(lr ** 3) / s ** 3) if s > 0 else 0.0,
          float(np.mean(lr ** 4) / s ** 4) if s > 0 else 0.0]
    up = float(np.mean([1.0 if v > 0 else 0.0 for v in lr[-20:]])) if len(lr) >= 2 else 0.5
    f += [up, float(np.max(np.abs(lr)))]
    return np.asarray(f, dtype=float)


def stat_matrix(X) -> np.ndarray:
    return np.stack([stat_features(x) for x in X])


def model_matrix(model, X, *, batch: int = 16) -> np.ndarray:
    """Frozen representation of ``X`` from the model (embeddings if it has them,
    else its forecast/quantile block — see ``TSFMModel.encode``)."""
    out = []
    for i in range(0, len(X), batch):
        out.append(np.atleast_2d(np.asarray(model.encode(list(X[i:i + batch])), dtype=float)))
    return np.concatenate(out, axis=0)


def make_classifier(kind: str = "logistic", seed: int = 0):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind in ("logistic", "logreg"):
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, C=1.0, random_state=seed))
    if kind in ("ridge",):
        from sklearn.linear_model import RidgeClassifier
        return make_pipeline(StandardScaler(), RidgeClassifier(random_state=seed))
    if kind in ("gbm", "gbm2", "hist"):
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(random_state=seed, max_depth=4,
                                              learning_rate=0.08, max_iter=200)
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return make_pipeline(StandardScaler(),
                             RandomForestClassifier(n_estimators=300, random_state=seed))
    raise ValueError(f"unknown probe classifier {kind!r}")


def folds(split: np.ndarray, n: int = 5, mode: str = "expanding"):
    """Yield ``(train_idx, test_idx)``.

    ``expanding`` (default for time-ordered windows): every test block is strictly
    after its training data — the only variant that is defensible for a series.
    ``stratified``: K-fold, out-of-fold w.r.t. the sample (window overlap means
    neighbouring windows are correlated; disclosed in the report, not hidden).
    """
    idx = np.arange(len(split))
    y_split = np.asarray(split)
    tr_idx = idx[y_split == 0]
    te_idx = idx[y_split == 1]
    if mode == "holdout" or len(tr_idx) < MIN_TRAIN or len(te_idx) == 0:
        if len(tr_idx) and len(te_idx):
            return [(tr_idx, te_idx)]
        # no split info: fall back to an expanding split
        k = max(1, len(idx) - MIN_TRAIN)
        return [(idx[:k], idx[k:])]
    if mode == "stratified":
        from sklearn.model_selection import StratifiedKFold
        return list(StratifiedKFold(n_splits=max(2, n), shuffle=True,
                                    random_state=0).split(idx, y_split))
    # expanding: cut the whole index range into n+1 blocks, train on the prefix
    blocks = np.array_split(idx, min(max(2, n + 1), max(2, len(idx) // MIN_TRAIN)))
    out = []
    for j in range(1, len(blocks)):
        out.append((np.concatenate(blocks[:j]), blocks[j]))
    return out
