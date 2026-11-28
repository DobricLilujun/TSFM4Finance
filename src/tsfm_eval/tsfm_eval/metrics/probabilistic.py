"""Quantile-based (probabilistic) metrics: pinball, CRPS, WQL, VaR/ES, coverage.

A benchmark row is only comparable if both sides compute the same functional. The arena and the
unified evaluator both need these, so the implementation lives here once; `tsfm4finance.core.metrics`
is a re-export shim over this package. The quantile grid is public because the shape of a
`(K, H)` vs `(H, K)` quantile array is resolved by matching `len(QUANTILE_LEVELS)` - changing the
grid silently changes what every CRPS row means.
"""
from __future__ import annotations

import numpy as np

#: the predictive quantile levels the benchmark protocol fixes for every model
QUANTILE_LEVELS: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def pinball(y: float | np.ndarray, q: float | np.ndarray, alpha: float) -> float:
    """Pinball (quantile) loss rho_alpha(y - q)."""
    r = np.asarray(y, dtype=float) - np.asarray(q, dtype=float)
    return float(np.mean(np.maximum(alpha * r, (alpha - 1.0) * r)))


def _as_levels_by_horizon(q: np.ndarray, n_levels: int, n_h: int, layout: str) -> np.ndarray | None:
    """Return the quantile matrix oriented (levels, horizons), or None if it cannot be that shape.

    The canonical public layout is **(horizons, levels)** - that is what `Pred.quantiles` is and how
    `bench.scoring` indexes it (`q[t, k]`). Guessing the orientation by matching `len(levels)` is
    wrong exactly when it matters: with the 9-level grid and a 9-step horizon the matrix is square,
    every guess "matches", and the metric is then computed across horizons instead of across levels
    while still returning a plausible number. So the layout is declared, not inferred; "auto" only
    resolves *unambiguous* shapes and refuses the square one.
    """
    Q = np.asarray(q, dtype=float)
    if Q.ndim != 2:
        return None
    square = Q.shape == (n_levels, n_levels) and n_h == n_levels
    if layout == "HK":
        Q = Q.T if Q.shape == (n_h, n_levels) else Q
    elif layout == "KH":
        pass
    elif layout == "auto":
        if square:
            return None                      # ambiguous by construction: do not guess
        if Q.shape[1] == n_levels and Q.shape[0] != n_levels:
            Q = Q.T
    if Q.shape != (n_levels, n_h):
        return None
    return Q


def crps_from_quantiles(truth: np.ndarray, quantiles: np.ndarray,
                        levels: list[float] | None = None, layout: str = "HK") -> float:
    """CRPS approximated from predictive quantiles.

    CRPS(F, y) = 2 * E_alpha[ rho_alpha(y - F^-1(alpha)) ]; the pinball loss is averaged over the
    supplied levels and multiplied by 2. `truth` is (H,), `quantiles` is (H, K) by default (see
    `_as_levels_by_horizon` for the layout contract); pass layout="KH" for a (K, H) matrix. A shape
    that cannot be resolved returns NaN rather than guessing, because a silently transposed quantile
    matrix still produces a number.
    """
    levels = levels or QUANTILE_LEVELS
    y = np.asarray(truth, dtype=float)
    Q = _as_levels_by_horizon(quantiles, len(levels), len(y), layout)
    if Q is None:
        return float("nan")
    tot = 0.0
    for k, a in enumerate(levels):
        tot += pinball(y, Q[k], a)
    return float(2.0 * tot / len(levels))


def wql(truth: np.ndarray, quantiles: np.ndarray, scale: float,
        levels: list[float] | None = None, layout: str = "HK") -> float:
    """Weighted quantile loss (GluonTS-style), normalized by `scale`."""
    c = crps_from_quantiles(truth, quantiles, levels, layout)
    if not np.isfinite(c) or scale == 0:
        return float("nan")
    return float(c / scale)


def var_es_from_quantiles(price_t: float, q_final: np.ndarray,
                          levels: list[float] | None = None) -> dict[str, float]:
    """VaR / Expected Shortfall of the h-step P&L from price quantiles.

    q_final: predicted price quantiles at the end of the horizon (same order as `levels`).
    Loss convention: a positive value is a loss of value.
    """
    levels = levels or QUANTILE_LEVELS
    q = {round(float(lv), 3): float(v) for lv, v in zip(levels, np.asarray(q_final, dtype=float))}
    lv_round = [round(float(lv), 3) for lv in levels]
    out: dict[str, float] = {}
    for conf in (0.9, 0.8):
        a = round(1.0 - conf, 3)
        if a not in q:
            continue
        out[f"var_{conf:.2f}"] = float(max(0.0, price_t - q[a]))
        tail = [q[lv] for lv in lv_round if lv <= a]
        out[f"es_{conf:.2f}"] = float(max(0.0, price_t - np.mean(tail)))
    return out


def quantile_coverage(truth: np.ndarray, quantiles: np.ndarray,
                      level: float = 0.1,
                      levels: list[float] | None = None, layout: str = "HK") -> float:
    """Fraction of outcomes falling below the `level` predictive quantile.

    A calibrated forecaster hits ~`level`; this is reported as a violation rate. Same layout contract
    as `crps_from_quantiles`: (H, K) unless told otherwise.
    """
    levels = levels or QUANTILE_LEVELS
    y = np.asarray(truth, dtype=float)
    Q = _as_levels_by_horizon(quantiles, len(levels), len(y), layout)
    if Q is None:
        return float("nan")
    hit = min(range(len(levels)), key=lambda i: abs(float(levels[i]) - float(level)))
    if abs(float(levels[hit]) - float(level)) > 1e-9:
        return float("nan")
    return float(np.mean(y < Q[hit]))


def normalized_difference(pred: np.ndarray, truth: np.ndarray,
                          history: np.ndarray | None = None,
                          season: int = 1) -> float:
    """MAE(model) / MAE(seasonal-naive). < 1 means better than seasonal naive.

    The denominator is a *scaled* quantity, so which history it is computed from is part of the
    measurement: pass the same split on both sides or the ratio is not comparable.
    """
    y = np.asarray(truth, dtype=float)
    m = float(np.mean(np.abs(np.asarray(pred, dtype=float) - y)))
    # seasonal naive: forecast each step by repeating the most recent season, i.e.
    # yhat[T+i] = y[T+i-season] for i in 1..H. Taking a *block of older history* instead
    # (h[-H-season:-season]) compares the truth against the wrong lag, and the ratio then moves
    # when irrelevant older data changes.
    if history is None or len(history) <= season:
        naive = np.roll(y, season)
        naive[:season] = y[0]
    else:
        h = np.asarray(history, dtype=float)
        last_season = h[-season:]
        naive = np.tile(last_season, int(np.ceil(len(y) / season)))[:len(y)]
    base = float(np.mean(np.abs(naive - y)))
    if base == 0:
        return float("nan")
    return m / base
