"""Trading-oriented metrics: cost-aware position backtests and information coefficients.

These are the quant-side metrics of the benchmark's finance spec (docs/02 §2.3): a model that
emits a *position* (or a signal turned into a position) is scored on what it would have earned
after transaction costs, not only on forecast error. They live in the canonical metrics package
because both the arena (`tsfm4finance.bench`) and the unified evaluator want them, and a second
copy is how the two trees start disagreeing about a number.

Costs are one-way basis points applied to turnover `|w_i - w_{i-1}|` with `w_{-1} = 0`. Returns
are log returns in, annualised means/vols out at `periods_per_year` (default 252 daily).
"""
from __future__ import annotations

import numpy as np

#: the cost grid every asset-return row is reported at (0 = frictionless, 10/30 bps realistic)
TRADING_COST_GRID: list[float] = [0.0, 10.0, 30.0]


def max_drawdown(returns: np.ndarray) -> float:
    """Max drawdown of a simple return stream (as a positive fraction)."""
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return float("nan")
    wealth = np.cumprod(1.0 + r)
    # the peak series starts at the initial wealth of 1.0: without that anchor a loss in the
    # first period of the window is compared against itself and reads as a 0% drawdown
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    dd = 1.0 - wealth / peak
    return float(np.max(dd))


def ic_signal(signal: np.ndarray, realized: np.ndarray) -> float:
    """Information Coefficient: Pearson corr(signal, forward return)."""
    s = np.asarray(signal, dtype=float)
    r = np.asarray(realized, dtype=float)
    m = np.isfinite(s) & np.isfinite(r)
    if m.sum() < 3 or np.std(s[m]) == 0 or np.std(r[m]) == 0:
        return float("nan")
    return float(np.corrcoef(s[m], r[m])[0, 1])


def trading_metrics(positions: np.ndarray, fwd_logret: np.ndarray,
                    cost_bps: float = 10.0, periods_per_year: int = 252,
                    signals: np.ndarray | None = None) -> dict[str, float]:
    """Cost-aware backtest of a position stream.

    positions[i] is the weight held over the interval that delivers fwd_logret[i] (log
    return). Turnover = |w_i - w_{i-1}| (w_{-1} = 0), charged one-way at cost_bps.
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

    # A position of weight w against an asset log return r does NOT earn w*r. A unit short into a
    # +10% month loses exactly 10% of the notional, while w*r would report -9.53%: log returns are
    # not additive across a signed weight. The portfolio is marked in simple space -
    # R_asset = expm1(r), R_p = w * R_asset - cost - and only then converted back to log space, so
    # "log return" columns stay log returns and shorts are signed correctly.
    r_simple = np.expm1(r)
    gross_simple = w * r_simple
    net_simple = gross_simple - cost
    # a position that loses everything (net_simple <= -1) has no log return; clamp at -1 and let the
    # caller see a wiped period rather than a nan that silently vanishes from an average
    log_gross = np.log1p(np.maximum(gross_simple, -1.0 + 1e-12))
    log_net = np.log1p(np.maximum(net_simple, -1.0 + 1e-12))
    gross, net = log_gross, log_net

    def _ann_mean(x):   # log-return stream -> annualized arithmetic-ish mean
        return float(np.mean(x) * periods_per_year)

    def _ann_vol(x):
        return float(np.std(x, ddof=1) * np.sqrt(periods_per_year)) if n > 1 else float("nan")

    def _sortino(x):
        dn = np.minimum(x, 0.0)
        dd = float(np.sqrt(np.mean(dn * dn)) * np.sqrt(periods_per_year))
        return float(_ann_mean(x) / dd) if dd > 0 else float("nan")

    def _sharpe(x):
        av, vv = _ann_mean(x), _ann_vol(x)
        return float(av / vv) if np.isfinite(vv) and vv > 0 else float("nan")

    simple_net = net_simple      # drawdown and cumulative reporting use simple returns
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
        # Sortino divides by the downside *deviation from the target* (zero here): root mean square
        # of the negative returns over every period. std() of only the negative rows is the wrong
        # denominator - a stream of identical losses has std 0 and reports an infinite ratio.
        "sortino_net": _sortino(net),
        "max_drawdown": max_drawdown(simple_net),
        "cum_return_net": float(np.prod(1.0 + simple_net) - 1.0),
        "cost_bps": float(cost_bps),
    }
    # t-stat of the mean net return (Harvey 2017 caveats about multiple testing still apply)
    m["t_stat_net"] = (float(np.mean(net) / (np.std(net, ddof=1) / np.sqrt(n)))
                       if n > 2 and np.std(net, ddof=1) > 0 else float("nan"))
    if signals is not None:
        m["ic"] = ic_signal(np.asarray(signals, dtype=float)[:n], r)
        try:
            from scipy.stats import spearmanr
            m["rank_ic"] = float(spearmanr(np.asarray(signals, dtype=float)[:n], r)[0])
        except Exception:                                          # noqa: BLE001
            import pandas as pd
            m["rank_ic"] = float(pd.Series(np.asarray(signals, dtype=float)[:n]).rank()
                                 .corr(pd.Series(r).rank()))
    m["calmar_net"] = (float(m["ann_return_net"] / m["max_drawdown"])
                       if np.isfinite(m.get("max_drawdown", float("nan")))
                       and m["max_drawdown"] > 0 else float("nan"))
    return m
