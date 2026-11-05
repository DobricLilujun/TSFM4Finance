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
        m["nd"] = normalized_difference(pred_pts, truth, history=hist)

        # probabilistic extras (spec §2): CRPS / WQL / VaR / ES / coverage
        if pred.quantiles is not None:
            Q = np.asarray(pred.quantiles, dtype=float)
            if Q.ndim == 2 and 9 in Q.shape:
                Q9 = Q[:9, :n] if Q.shape[0] >= 9 else Q.T[:9, :n]
                m["crps"] = crps_from_quantiles(truth, Q9)
                scale = (float(np.mean(np.abs(np.diff(hist)))) if hist is not None
                         and len(hist) > 1 else float(np.mean(np.abs(truth))))
                m["wql"] = wql(truth, Q9, scale)
                last = [Q9[k, -1] for k in range(9)]
                m.update({f"fc_{k}": float(v) for k, v in zip(
                    ["var_0.90", "es_0.90", "var_0.80", "es_0.80"],
                    list(var_es_from_quantiles(
                        float(pred_pts[0] if len(pred_pts) else truth[0]), last).values()))})
                m["violation_rate"] = quantile_coverage(truth, Q9, 0.1)
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

# --------------------------------------------------------------------------- #
# Probabilistic metrics (spec §2: CRPS) — TimesFM / Chronos return quantiles
# --------------------------------------------------------------------------- #
QUANTILE_LEVELS: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def pinball(y: float | np.ndarray, q: float | np.ndarray, alpha: float) -> float:
    """Pinball (quantile) loss rho_alpha(y - q)."""
    r = np.asarray(y, dtype=float) - np.asarray(q, dtype=float)
    return float(np.mean(np.maximum(alpha * r, (alpha - 1.0) * r)))


def crps_from_quantiles(truth: np.ndarray, quantiles: np.ndarray,
                        levels: list[float] | None = None) -> float:
    """CRPS approximated from predictive quantiles.

    CRPS(F, y) = 2 * E_alpha[ rho_alpha(y - F^-1(alpha)) ]; we average the pinball
    loss over the supplied levels and multiply by 2. `truth` (H,), `quantiles`
    (K, H) or (H, K) — detected by matching `len(levels)`.
    """
    levels = levels or QUANTILE_LEVELS
    Q = np.asarray(quantiles, dtype=float)
    y = np.asarray(truth, dtype=float)
    if Q.ndim != 2:
        return float("nan")
    if Q.shape[1] == len(levels) and Q.shape[0] != len(levels):
        Q = Q.T                              # -> (K, H)
    if Q.shape[0] != len(levels) or Q.shape[1] != len(y):
        return float("nan")
    tot = 0.0
    for k, a in enumerate(levels):
        tot += pinball(y, Q[k], a)
    return float(2.0 * tot / len(levels))


def wql(truth: np.ndarray, quantiles: np.ndarray, scale: float,
        levels: list[float] | None = None) -> float:
    """Weighted quantile loss (GluonTS-style), normalized by `scale`."""
    c = crps_from_quantiles(truth, quantiles, levels)
    if not np.isfinite(c) or scale == 0:
        return float("nan")
    return float(c / scale)


def var_es_from_quantiles(price_t: float, q_final: np.ndarray,
                          levels: list[float] | None = None) -> dict[str, float]:
    """VaR / Expected Shortfall of the h-step P&L from price quantiles.

    q_final: predicted price quantiles at the end of the horizon (same order as
    `levels`). Loss convention: positive loss = loss of value.
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
                      levels: list[float] | None = None) -> float:
    """Fraction of outcomes falling below the `level` predictive quantile.

    A calibrated forecaster hits ~`level`. Reported as `violation_rate`."""
    levels = levels or QUANTILE_LEVELS
    Q = np.asarray(quantiles, dtype=float)
    if Q.shape[1] == len(levels) and Q.shape[0] != len(levels):
        Q = Q.T
    if level not in levels or Q.ndim != 2:
        return float("nan")
    row = Q[levels.index(level)]
    return float(np.mean(np.asarray(truth, dtype=float) < row))


def normalized_difference(pred: np.ndarray, truth: np.ndarray,
                          history: np.ndarray | None = None,
                          season: int = 1) -> float:
    """MAE(model) / MAE(seasonal-naive). < 1 means better than seasonal naive."""
    y = np.asarray(truth, dtype=float)
    m = float(np.mean(np.abs(np.asarray(pred, dtype=float) - y)))
    if history is None or len(history) <= season:
        naive = np.roll(y, season)
        naive[:season] = y[0]
    else:
        h = np.asarray(history, dtype=float)
        naive = h[-len(y) - season:-season] if len(h) >= len(y) + season else np.roll(y, season)
    base = float(np.mean(np.abs(naive - y)))
    if base == 0:
        return float("nan")
    return m / base


# --------------------------------------------------------------------------- #
# Trading metrics (spec §2.3)
# --------------------------------------------------------------------------- #
TRADING_COST_GRID: list[float] = [0.0, 10.0, 30.0]


def ic_signal(signal: np.ndarray, realized: np.ndarray) -> float:
    """Information Coefficient: Pearson corr(signal, forward return)."""
    s = np.asarray(signal, dtype=float); r = np.asarray(realized, dtype=float)
    m = np.isfinite(s) & np.isfinite(r)
    if m.sum() < 3 or np.std(s[m]) == 0 or np.std(r[m]) == 0:
        return float("nan")
    return float(np.corrcoef(s[m], r[m])[0, 1])


def max_drawdown(returns: np.ndarray) -> float:
    """Max drawdown of a simple return stream (as a positive fraction)."""
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return float("nan")
    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(wealth)
    dd = 1.0 - wealth / peak
    return float(np.max(dd))


def trading_metrics(positions: np.ndarray, fwd_logret: np.ndarray,
                    cost_bps: float = 10.0, periods_per_year: int = 252,
                    signals: np.ndarray | None = None) -> dict[str, float]:
    """Cost-aware backtest of a position stream.

    positions[i] is the weight held over the interval that delivers
    fwd_logret[i] (log return). Turnover = |w_i - w_{i-1}| (w_{-1} = 0), charged
    one-way at cost_bps. Returns the quant-trading metric block of spec §2.3.
    """
    w = np.asarray(positions, dtype=float)
    r = np.asarray(fwd_logret, dtype=float)
    n = min(len(w), len(r))
    w, r = w[:n], r[:n]
    if n == 0:
        return {}
    prev = np.concatenate([[0.0], w[:-1]])
    turnover = np.abs(w - prev)
    cost = turnover * (cost_bps / 1e4)
    gross = w * r
    net = gross - cost

    def _ann_mean(x):   # log-return stream -> annualized arithmetic-ish mean
        return float(np.mean(x) * periods_per_year)

    def _ann_vol(x):
        return float(np.std(x, ddof=1) * np.sqrt(periods_per_year)) if n > 1 else float("nan")

    def _sharpe(x):
        av, vv = _ann_mean(x), _ann_vol(x)
        return float(av / vv) if np.isfinite(vv) and vv > 0 else float("nan")

    # simple (non-log) returns for drawdown / cumulative reporting
    simple_net = np.expm1(net)
    m: dict[str, float] = {
        "n_bets": float(n),
        "hit_rate": float(np.mean(np.sign(w) == np.sign(r))),
        "avg_position": float(np.mean(np.abs(w))),
        "turnover_total": float(np.sum(turnover) / max(1, n)),
        "ann_return_gross": _ann_mean(gross),
        "ann_return_net": _ann_mean(net),
        "ann_vol": _ann_vol(net),
        "sharpe_gross": _sharpe(gross),
        "sharpe_net": _sharpe(net),
        "sortino_net": (float(_ann_mean(net) /
                              (np.std(net[net < 0], ddof=1) * np.sqrt(periods_per_year)))
                        if np.any(net < 0) else float("nan")),
        "max_drawdown": max_drawdown(simple_net),
        "cum_return_net": float(np.expm1(np.sum(net))),
        "cost_bps": float(cost_bps),
    }
    # t-stat of the mean net return (significance; Harvey 2017 caveats apply)
    m["t_stat_net"] = (float(np.mean(net) / (np.std(net, ddof=1) / np.sqrt(n)))
                       if n > 2 and np.std(net, ddof=1) > 0 else float("nan"))
    if signals is not None:
        m["ic"] = ic_signal(np.asarray(signals, dtype=float)[:n], r)
        try:
            from scipy.stats import spearmanr
            m["rank_ic"] = float(spearmanr(np.asarray(signals, dtype=float)[:n], r)[0])
        except Exception:
            pr = pd.Series(np.asarray(signals, dtype=float)[:n]).rank()
            rr = pd.Series(r).rank()
            m["rank_ic"] = float(pr.corr(rr))
    m["calmar_net"] = (float(m["ann_return_net"] / m["max_drawdown"])
                       if np.isfinite(m.get("max_drawdown", float("nan")))
                       and m["max_drawdown"] > 0 else float("nan"))
    return m


def _score_trading(m: dict[str, float]) -> float:
    """Composite trading score: net Sharpe (capped) + IC + hit rate."""
    sh = m.get("sharpe_net", 0.0)
    s = 0.5 * _clip01((sh + 1.0) / 4.0)          # -1..3 Sharpe -> 0..1
    ic = m.get("ic", 0.0)
    s += 0.3 * _clip01((ic + 0.1) / 0.3)         # -0.1..0.2 IC -> 0..1
    s += 0.2 * m.get("hit_rate", 0.5)
    return round(_clip01(s), 4)


# register the trading score after its definition (defined further below)
_COMPOSITE[TaskType.TRADING] = _score_trading
