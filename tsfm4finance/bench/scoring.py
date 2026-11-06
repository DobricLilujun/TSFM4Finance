"""Unified forecast scoring: statistics + the finance-specific layer, in one place.

Both the sweep worker and the smoke test call ``score_forecast`` so the numbers a
cell stores are always produced by the same code path.

Design decisions worth knowing when reading a results row:

* MASE scale comes from the **train split only** (per series), never from the
  evaluation window - otherwise the denominator leaks.
* Every model is compared to the persistence forecast *on the identical windows*
  (NDp and the Diebold-Mariano test), so "beats naive" is a paired statement.
* Volatility calibration is derived from the quantile spread
  (sigma_hat = (q90 - q10) / 2.563) and regressed against realised variance in the
  Mincer-Zarkin sense; models without quantiles get `null` there instead of a
  fabricated number.
* VaR backtests (Kupiec, Christoffersen) use the 10% lower quantile of the *last*
  step of each window, which is the horizon a risk desk would actually hold.
"""
from __future__ import annotations

import numpy as np

from .engines.base import LEVELS
# NB: imported as a module, not a name list.  A `from .finance_metrics import (…)`
# name list fails intermittently in this package (stale-module resolution inside
# `python -m` runs), and a module import is unambiguous and cheaper to read.
from . import finance_metrics as FM

Z = {"0.10": 1.2816, "0.90": 1.2816}


def _mase_scale(train_hist: np.ndarray) -> float:
    d = np.abs(np.diff(train_hist))
    s = float(np.mean(d)) if d.size else float("nan")
    return s if s > 0 else float("nan")


def _implied_sigma(q: np.ndarray) -> float:
    """sigma implied by the 10-90 quantile band (normal reference)."""
    try:
        return max(0.0, (float(q[8]) - float(q[0])) / (2 * 1.2816))
    except Exception:                                       # noqa: BLE001
        return float("nan")


def per_window_metrics(wins, pred, *, train_hist: np.ndarray, levels=LEVELS) -> list[dict]:
    from tsfm4finance.core import metrics as M
    H = len(wins[0].truth)
    rows = []
    for i, w in enumerate(wins):
        p = np.asarray(pred.point[i], float)[:H]
        y = np.asarray(w.truth, float)[:H]
        row = {
            "rmse": M.rmse(p, y), "mae": M.mae(p, y),
            "mape": M.mape(p, y), "smape": M.smape(p, y),
            "mase": M.mase(p, y, history=train_hist),
            "dir_acc": M.directional_accuracy(p, y),
        }
        if pred.quantiles is not None:
            q = np.asarray(pred.quantiles[i], float)[:H]          # (H, K)
            row["crps"] = M.crps_from_quantiles(y, q, levels=list(levels))
            row["wql"] = M.wql(y, q, scale=max(np.nanstd(np.concatenate([w.context, y])), 1e-9),
                               levels=list(levels))
            row["pinball_mean"] = float(np.mean([M.pinball(y[t], q[t, k], levels[k])
                                                 for t in range(H) for k in range(len(levels))]))
            lower = M.var_es_from_quantiles(float(w.context[-1]), q[-1], levels=list(levels)) \
                if hasattr(M, "var_es_from_quantiles") else {}
            row["var90"] = float(q[-1, 0])
            row["es90"] = float(np.mean(q[-1, :3]))
            row["exceed_10"] = float(y[-1] < q[-1, 0])
            row["band_10_exceeds"] = float(np.mean(y < q[:, 0]))
            row["band_10_hits"] = float(np.mean((y >= q[:, 0]) & (y <= q[:, 8])))
            row["sigma_hat"] = float(np.nanmean([_implied_sigma(q[t]) for t in range(H)]))
            row["sigma_real"] = float(np.std(y))
        rows.append(row)
    return rows


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
    out: dict = {}
    for k in keys:
        vals = [r[k] for r in rows if np.isfinite(r[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
        out[f"{k}__std"] = float(np.std(vals)) if len(vals) > 1 else float("nan")
    out["n_origins"] = len(rows)
    return out


def finance_layer(wins, pred, *, baseline_point: np.ndarray | None, horizon: int,
                  scale: float, levels=LEVELS) -> dict:
    """Paired significance, calibration and VaR backtests."""
    out: dict = {}
    Y = np.concatenate([np.asarray(w.truth, float) for w in wins])
    P = np.asarray(pred.point, float)[:, :len(wins[0].truth)].reshape(-1)
    if baseline_point is not None:
        B = np.asarray(baseline_point, float).reshape(-1)
        if B.shape == P.shape:
            dm = FM.diebold_mariano(P - Y, B - Y, horizon=horizon)
            out["dm_t_vs_naive"], out["dm_p_vs_naive"] = dm["dm_t"], dm["p_value"]
            out["dm_favours"] = dm["favours"]
            out["mz_t_vs_naive"] = dm["mz_t"]
    if pred.quantiles is not None:
        Q = np.asarray(pred.quantiles, float)[:, :len(wins[0].truth)]
        Qf = Q.reshape(-1, len(levels))
        Yf = Y.reshape(-1)
        out.update(FM.pit_diagnostics(Yf, Qf, list(levels)))
        out.update(FM.coverage_by_level(Yf, Qf, list(levels)))
        # VaR backtest on the last step of each window (the held horizon)
        lower = Q[:, -1, 0]
        truth_last = np.array([w.truth[-1] for w in wins])
        out.update(FM.kupiec_pof(len(truth_last), int(np.sum(truth_last < lower)), 0.10))
        out.update(FM.christoffersen_indep((truth_last < lower).astype(int)))
        # volatility calibration from the quantile band
        sh = np.array([np.nanmean([_implied_sigma(Q[i, t]) for t in range(Q.shape[1])])
                       for i in range(len(wins))])
        sr = np.array([np.std(np.asarray(w.truth, float)) for w in wins])
        m = np.isfinite(sh) & np.isfinite(sr) & (sh > 0)
        if m.sum() >= 5:
            out["vol_zscore_ratio"] = FM.vol_zscore_ratio(sh[m], sr[m])
            out.update({f"mzv_{k}": v for k, v in
                        FM.mz_variance_regression(sh[m] ** 2, sr[m] ** 2).items()})
            out["qlike_vol"] = FM.qlike(sh[m] ** 2, sr[m] ** 2)
    return out


def score_forecast(wins, pred, *, info: dict, train_hist: np.ndarray, horizon: int,
                   baseline_point: np.ndarray | None = None, model_key: str = "",
                   series: str = "", adaptation: str = "zero") -> dict:
    scale = _mase_scale(train_hist)
    rows = per_window_metrics(wins, pred, train_hist=train_hist)
    agg = aggregate(rows)
    cell = {
        "stage": "forecast", "series": series, "model": model_key, "adaptation": adaptation,
        "horizon": int(horizon), "span": info.get("span"), "n_origins": agg.get("n_origins"),
        "coverage": info.get("coverage"), "no_leakage": info.get("no_leakage"),
        "has_quantiles": pred.quantiles is not None,
        "aggregate": agg,
        "per_window": rows,
    }
    cell["finance"] = finance_layer(wins, pred, baseline_point=baseline_point,
                                    horizon=horizon, scale=scale)
    naive_mase = agg.get("mase")
    if baseline_point is not None and np.isfinite(naive_mase):
        # NDp: MAE(model)/MAE(persistence) on identical windows
        B = np.asarray(baseline_point, float).reshape(len(wins), -1)
        mae_naive = float(np.mean([np.mean(np.abs(B[i] - w.truth)) for i, w in enumerate(wins)]))
        cell["ndp"] = float(agg["mae"] / mae_naive) if mae_naive > 0 else float("nan")
    return cell
