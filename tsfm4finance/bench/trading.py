"""The `trading` task (spec §2.3): forecast -> position -> cost-aware backtest.

Protocol (fixed, so numbers are comparable across models — ml4trading ch.16 requires the
protocol to be explicit):

  signal timing : decision at the close of day t, using prices <= t only;
                  P&L accrues over (t, t+h].
  rebalancing   : non-overlapping origins, stride = h (no double-counted P&L).
  position      : "sign"       w = sign(s)            (long/short, unit size)
                  "voltarget"  w = clip(s / sigma_hat, -1, +1)
  signal s      : predicted h-step log return log(yhat_{t+h} / P_t)
  sigma_hat     : predicted h-step volatility from the 0.1-0.9 predictive band:
                  (q90 - q10) / (2 * 1.2816) / P_t
  cost model    : one-way cost_bps on |change in position|; reported at 0/10/30 bps
                  (cost sensitivity is a protocol requirement).
  constraints   |w| <= 1, single asset, no leverage, no short-cost/borrow modelling.
"""
from __future__ import annotations

import numpy as np

from tsfm4finance.core.metrics import trading_metrics, TRADING_COST_GRID

Z90 = 1.2816


def signal_from_forecast(context: np.ndarray, point: np.ndarray,
                         quantiles: np.ndarray | None, h: int) -> tuple[float, float]:
    """(predicted h-step log return, predicted h-step volatility)."""
    p0 = float(context[-1])
    pt = np.asarray(point, dtype=float)
    s = float(np.log(max(pt[-1], 1e-9) / max(p0, 1e-9)))
    sig = np.nan
    if quantiles is not None:
        Q = np.asarray(quantiles, dtype=float)            # (h, 9)
        if Q.ndim == 2 and Q.shape[1] >= 9:
            band = float(Q[-1, 8] - Q[-1, 0])             # q90 - q10 at the horizon end
            sig = band / (2.0 * Z90) / max(p0, 1e-9)
    if not np.isfinite(sig) or sig <= 0:
        lr = np.r_[0.0, np.diff(np.log(np.maximum(context, 1e-12)))]
        sig = float(np.std(lr[-min(63, len(lr)):], ddof=1) * np.sqrt(h)) if len(lr) > 2 else 1e-3
    return s, float(max(sig, 1e-8))


def positions(signals: np.ndarray, sigmas: np.ndarray, mode: str = "sign") -> np.ndarray:
    s = np.asarray(signals, dtype=float)
    if mode == "sign":
        return np.sign(s)
    if mode == "voltarget":
        sg = np.asarray(sigmas, dtype=float)
        return np.clip(s / np.maximum(sg, 1e-8), -1.0, 1.0)
    raise ValueError(f"unknown position mode {mode!r}")


def momentum_positions(close: np.ndarray, origins: list[int], h: int) -> np.ndarray:
    """Benchmark: sign of the trailing h-day return (classic short-horizon momentum)."""
    c = np.asarray(close, dtype=float)
    out = []
    for o in origins:
        if o - h < 0:
            out.append(0.0)
        else:
            out.append(float(np.sign(np.log(c[o - 1] / c[o - 1 - h]))))
    return np.array(out)


def trade_eval(contexts: list[np.ndarray], points: np.ndarray,
               quantiles: list[np.ndarray] | None, origins: list[int],
               close: np.ndarray, fwd_logret: np.ndarray, h: int,
               cost_grid: list[float] | None = None) -> dict:
    """Evaluate a TSFM's trading performance at one horizon (both position rules)."""
    cost_grid = cost_grid or TRADING_COST_GRID
    pts = np.asarray(points, dtype=float)
    sig_list, sg_list = [], []
    for i, ctx in enumerate(contexts):
        q = quantiles[i] if quantiles is not None else None
        s_i, sg_i = signal_from_forecast(ctx, pts[i], q, h)
        sig_list.append(s_i); sg_list.append(sg_i)
    sig = np.array(sig_list)
    sgs = np.array(sg_list)
    res: dict = {}
    for mode in ("sign", "voltarget"):
        pos = positions(sig, sgs, mode)
        block: dict = {}
        for cb in cost_grid:
            m = trading_metrics(pos, fwd_logret, cost_bps=cb, signals=sig)
            block[f"{cb:.0f}bps"] = m
        block["cost_sensitivity_sharpe"] = [block[f"{cb:.0f}bps"]["sharpe_net"] for cb in cost_grid]
        block["mean_signal"] = float(np.mean(sig))
        block["mean_sigma_hat"] = float(np.mean(sgs))
        res[mode] = block
    return res


def benchmark_trades(close: np.ndarray, origins: list[int], fwd_logret: np.ndarray, h: int,
                     cost_grid: list[float] | None = None) -> dict:
    """buy&hold, momentum and random benchmarks for the same trade calendar."""
    import numpy as _np
    cost_grid = cost_grid or TRADING_COST_GRID
    out: dict = {}
    n = len(origins)
    rules = {
        "buy_and_hold": _np.ones(n),
        "momentum": momentum_positions(close, origins, h),
        "random": _np.random.RandomState(0).choice([-1.0, 1.0], size=n),
    }
    for name, pos in rules.items():
        block = {}
        for cb in cost_grid:
            block[f"{cb:.0f}bps"] = trading_metrics(pos, fwd_logret, cost_bps=cb)
        block["cost_sensitivity_sharpe"] = [block[f"{cb:.0f}bps"]["sharpe_net"] for cb in cost_grid]
        out[name] = block
    return out
