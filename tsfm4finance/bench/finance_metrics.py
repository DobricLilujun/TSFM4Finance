"""Finance-specific evaluation metrics that the statistical set does not cover.

Why these are in the benchmark (the argument the survey's §4.2 makes):

1. *Paired significance* (Diebold-Mariano / MZ, with HAC).  In finance the gaps we
   care about are ~0.05-0.15 MASE on 5-30 windows.  Without a paired loss-difference
   test, a multi-model table is a ranking of noise.  Because our protocol uses
   non-overlapping h-step windows, the loss differences are h-dependent, so the
   standard Newey-West (h-1)-lag variance estimator is the right one.
2. *Multiple-testing control* (Deflated Sharpe Ratio, stationary-bootstrap SPA,
   Benjamini-Hochberg).  A sweep over ~15 models x 12+ series x 5 horizons x 2
   position rules is a search over hundreds of Sharpe ratios; the maximum of that
   search is not a strategy.  DSR/SPA convert "best cell" into a statement about
   the search, and BH bounds the false-discovery rate of the accuracy claims.
3. *Loss-function honesty for volatility* (QLIKE, Mincer-Zarkin variance regression).
   Risk use is scale-sensitive: MAE on volatility treats a 10% miss at 2% vol as
   equal to a 10% miss at 40% vol.  QLIKE is robust to both outliers and
   forecast-zero degeneracy, and the MZ variance regression tests whether variance
   is *calibrated*, not just accurate.
4. *Distributional calibration* (PIT, per-level coverage).  Point error says nothing
   about whether the predictive distribution is usable for VaR/ES.
5. *Regulatory-style backtests* (Kupiec POF, Christoffersen independence).  VaR
   exceedance counts and clustering are what a risk desk audits; a model can have a
   good CRPS and still produce clustered, unusable exceedances.
6. *Rare-event ranking* (PR-AUC, F-beta).  Extreme-move base rates are 1-3%; ROC-AUC
   is flattering there and macro-F1 is brittle, so average precision is reported.
7. *Economic shape* (Ulcer index, drawdown duration, market capture, cost slope).
   Sharpe alone hides drawdown length, benchmark dependence and cost fragility.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

try:                                    # scipy is a hard dep of the bench extra
    from scipy import stats as _sps
except Exception:                       # pragma: no cover
    _sps = None


# --------------------------------------------------------------------------- #
# 1. paired significance
# --------------------------------------------------------------------------- #
def _loss_diffs(e_a: np.ndarray, e_b: np.ndarray, loss: str = "sq") -> np.ndarray:
    a, b = np.asarray(e_a, float).ravel(), np.asarray(e_b, float).ravel()
    if loss == "sq":
        la, lb = a**2, b**2
    elif loss in ("abs", "ae"):
        la, lb = np.abs(a), np.abs(b)
    else:
        raise ValueError(f"unknown loss {loss!r}")
    return la - lb                       # >0 -> model A worse


def hac_variance(x: np.ndarray, lags: int) -> float:
    """Long-run variance of the mean of x (Newey-West, Bartlett kernel)."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 3:
        return float("nan")
    xc = x - x.mean()
    g0 = float(np.dot(xc, xc) / n)
    s = g0
    for k in range(1, min(lags, n - 1) + 1):
        gk = float(np.dot(xc[k:], xc[:-k]) / n)
        s += 2.0 * (1.0 - k / (lags + 1.0)) * gk
    return s


def diebold_mariano(err_a: np.ndarray, err_b: np.ndarray, horizon: int = 1,
                    loss: str = "sq") -> dict:
    """DM(1995) t-statistic on the loss difference (B1 heteroskedastic version).

    Positive t => model B is better.  Reported together with the MZ (Harvey,
    Leybourne & Newbold 1997) small-sample correction, which for h-step errors is
    t / sqrt(h) -- financial h-step errors are negatively autocorrelated, so the
    uncorrected DM statistic over-rejects.
    """
    d = _loss_diffs(err_a, err_b, loss)
    n = len(d)
    if n < max(5, horizon + 2):
        return {"dm_t": float("nan"), "mz_t": float("nan"), "p_value": float("nan"),
                "n": n, "favours": "inconclusive"}
    var = hac_variance(d, lags=max(0, horizon - 1))
    if not np.isfinite(var) or var <= 0:
        # identical error series: the loss difference is a constant, so a tie is the
        # honest answer rather than an undefined statistic.
        if np.isfinite(var) and abs(float(np.mean(d))) < 1e-12:
            return {"dm_t": 0.0, "mz_t": 0.0, "p_value": 1.0, "p_value_mz": 1.0,
                    "n": n, "favours": "tie", "loss": loss,
                    "hac_lags": max(0, horizon - 1)}
        return {"dm_t": float("nan"), "mz_t": float("nan"), "p_value": float("nan"),
                "n": n, "favours": "inconclusive"}
    t = float(np.mean(d) / math.sqrt(var / n))
    mz = t / math.sqrt(horizon)
    p = 2.0 * (1.0 - _norm_cdf(abs(t))) if _sps is None else 2.0 * _sps.t.sf(abs(t), df=n - 1)
    pmz = (2.0 * (1.0 - _norm_cdf(abs(mz)))) if _sps is None else 2.0 * _sps.t.sf(abs(mz), df=n - 1)
    favours = "B" if t > 0 else ("A" if t < 0 else "tie")
    return {"dm_t": t, "mz_t": float(mz), "p_value": float(p), "p_value_mz": float(pmz),
            "n": n, "favours": favours, "loss": loss, "hac_lags": max(0, horizon - 1)}


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# --------------------------------------------------------------------------- #
# 2. multiple-testing control for the trading grid
# --------------------------------------------------------------------------- #
def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return float("nan")
    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(r.mean() / sd * math.sqrt(periods_per_year))


def deflated_sharpe_ratio(returns: np.ndarray, n_trials: int,
                          periods_per_year: int = 252) -> dict:
    """Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio, annualised.

    ``n_trials`` is the number of strategy configurations actually searched (our
    whole trading grid), so the null Sharpe is the *expected maximum* of that
    many noisy draws rather than zero.
    """
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    T = len(r)
    if T < 5 or n_trials < 1:
        return {"sr": float("nan"), "sr0": float("nan"), "dsr": float("nan"),
                "p_value": float("nan"), "n_trials": n_trials, "T": T}
    sr = r.mean() / r.std(ddof=1)
    g3 = float(((r - r.mean()) ** 3).mean() / r.std() ** 3)
    g4 = float(((r - r.mean()) ** 4).mean() / r.std() ** 4)
    # E[max SR] over n_trials independent draws of a standard normal (Bailey &
    # Lopez de Prado 2014, eq. 5-6); n_trials=1 -> null Sharpe 0 (plain t-test).
    gamma = 0.5772156649015329
    if n_trials > 1:
        z1 = _norm_ppf(1.0 - 1.0 / n_trials)
        z2 = _norm_ppf(1.0 - 1.0 / (2.0 * n_trials))
        sr0 = math.sqrt(max(0.0, (1.0 - gamma) * z1**2 + gamma * z2**2))
    else:
        sr0 = 0.0
    denom = math.sqrt(max(1e-12, 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2))
    dsr = _norm_cdf((sr - sr0) * math.sqrt(T - 1) / denom)
    return {"sr_annual": sr * math.sqrt(periods_per_year), "sr0": sr0,
            "sr_per_period": sr, "dsr": float(dsr), "p_value": float(1.0 - dsr),
            "n_trials": int(n_trials), "T": T, "skew": g3, "kurtosis": g4}


def _norm_ppf(p: float) -> float:
    if _sps is not None:
        return float(_sps.norm.ppf(p))
    # Acklam's inverse normal CDF approximation
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671211905052e-01, 2.445134137142996e+00,
         3.754408664644747e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return ((c[0]*q**5+c[1]*q**4+c[2]*q**3+c[3]*q**2+c[4]*q+c[5]) /
                (d[0]*q**5+d[1]*q**4+d[2]*q**3+d[3]*q**2+1))
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -((c[0]*q**5+c[1]*q**4+c[2]*q**3+c[3]*q**2+c[4]*q+c[5]) /
                 (d[0]*q**5+d[1]*q**4+d[2]*q**3+d[3]*q**2+1))
    q = p - 0.5
    r = q * q
    return (((a[0]*r**5+a[1]*r**4+a[2]*r**3+a[3]*r**2+a[4]*r+a[5]) * q /
             (b[0]*r**4+b[1]*r**3+b[2]*r**2+b[3]*r+b[4])))


def stationary_bootstrap_spa(strat: np.ndarray, bench: np.ndarray, n_boot: int = 2000,
                            avg_block: int = 12, seed: int = 0) -> dict:
    """Hansen's SPA / White's Reality Check via a stationary bootstrap.

    Tests H0: no column of ``strat`` beats ``bench`` in mean.  Returns the SPA
    p-value and the max-stat bootstrap critical value; the p-value already folds in
    the search over columns, which is the whole point for a model sweep.
    """
    # Hansen's SPA: S is (T, k) - rows are periods, columns are strategies.
    b = np.asarray(bench, float).ravel()
    S = np.asarray(strat, float)
    if S.ndim == 1:
        S = S[:, None]                                  # (T,) -> (T,1)
    if S.shape[0] != b.size and S.shape[1] == b.size:
        S = S.T                                         # accept (k,T) input too
    keep = np.isfinite(b) & np.all(np.isfinite(S), axis=1)
    S, b = S[keep], b[keep]
    T, k = S.shape
    if T < 8 or k == 0:
        return {"spa_p": float("nan"), "max_stat": float("nan"), "n_boot": n_boot}
    d = S - b[:, None]                                  # per-period loss/return difference
    obs = d.mean(axis=0)
    stat = float(np.max(obs))
    sd = d.std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    t_obs = obs / (sd / math.sqrt(T))
    rng = np.random.default_rng(seed)
    p_keep = 1.0 / avg_block
    max_boot = np.empty(n_boot)
    for bi in range(n_boot):
        idx = np.empty(T, dtype=int)
        i = 0
        while i < T:
            j = int(rng.integers(0, T))
            idx[i] = j
            i += 1
            while i < T and rng.random() > p_keep:
                idx[i] = (idx[i - 1] + 1) % T
                i += 1
        db = d[idx, :]
        mb = db.mean(axis=0) - obs            # centre: H0 says no strategy beats bench
        tb = mb / (sd / math.sqrt(T))
        max_boot[bi] = np.max(tb)
    p = float((np.sum(max_boot >= np.max(t_obs)) + 1.0) / (n_boot + 1.0))
    crit = float(np.quantile(max_boot, 0.95))
    return {"spa_p": p, "max_stat": stat, "max_t": float(np.max(t_obs)),
            "crit95": crit, "n_boot": int(n_boot), "n_strategies": int(k),
            "avg_block": int(avg_block)}


def benjamini_hochberg(pvals: np.ndarray, q: float = 0.10) -> dict:
    """BH step-up: which of a family of accuracy claims survive FDR control."""
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    p_ok, idx = p[ok], np.where(ok)[0]
    m = len(p_ok)
    if m == 0:
        return {"rejected": [], "p_adj": [], "q": q, "m": 0}
    order = np.argsort(p_ok)
    sorted_p = p_ok[order]
    adj_sorted = np.minimum.accumulate((sorted_p * m / np.arange(1, m + 1))[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0, 1)
    out_adj = np.full(m, np.nan)
    out_adj[order] = adj_sorted
    rejected = [int(idx[order[r]]) for r in range(m) if adj_sorted[r] <= q]
    return {"rejected": rejected, "p_adj": [float(x) for x in out_adj], "q": q, "m": m}


# --------------------------------------------------------------------------- #
# 3. volatility: QLIKE + Mincer-Zarkin variance calibration
# --------------------------------------------------------------------------- #
def qlike(pred_var: np.ndarray, real_var: np.ndarray, eps: float = 1e-12) -> float:
    """QLIKE = log sigma^2_hat + sigma^2 / sigma^2_hat (Patton 2011). Scale-free in
    the ratio sense, robust to outliers, and undefined-free for near-zero forecasts."""
    pv = np.asarray(pred_var, float)
    rv = np.asarray(real_var, float)
    m = np.isfinite(pv) & np.isfinite(rv) & (pv > eps)
    if m.sum() < 3:
        return float("nan")
    return float(np.mean(np.log(pv[m]) + rv[m] / pv[m]))


def mz_variance_regression(pred_var: np.ndarray, real_var: np.ndarray) -> dict:
    """Mincer-Zarkin variance calibration: real_var = a + b * pred_var.
    Ideal: a=0, b=1.  a>0 -> under-forecasting risk; b<1 -> over-confident swings."""
    x = np.asarray(pred_var, float)
    y = np.asarray(real_var, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 5 or x.std() == 0:
        return {"intercept": float("nan"), "slope": float("nan"), "r2": float("nan"),
                "n": int(len(x))}
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(x) - 2
    s2 = float(resid @ resid) / max(1, dof)
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"intercept": float(beta[0]), "intercept_t": float(beta[0] / se[0]),
            "slope": float(beta[1]), "slope_t": float(beta[1] / se[1]),
            "slope_se": float(se[1]),
            "r2": float(1.0 - (resid @ resid) / ss_tot) if ss_tot > 0 else float("nan"),
            "n": int(len(x))}


def vol_zscore_ratio(pred_vol: np.ndarray, real_vol: np.ndarray) -> float:
    """Mean z-score of realised vol under the predicted vol; 0 = calibrated,
    negative = over-forecasting risk (the failure mode that matters for capital)."""
    p = np.asarray(pred_vol, float)
    r = np.asarray(real_vol, float)
    m = np.isfinite(p) & np.isfinite(r) & (p > 0)
    if m.sum() < 3:
        return float("nan")
    return float(np.mean((r[m] - p[m]) / p[m]))


# --------------------------------------------------------------------------- #
# 4-5. distributional calibration and VaR backtests
# --------------------------------------------------------------------------- #
def pit_values(truth: np.ndarray, quantiles: np.ndarray,
               levels: list[float]) -> np.ndarray:
    """Probability-integral transform of the truth under the predicted CDF, by
    linear interpolation across the reported quantile grid. Uniform(0,1) if the
    predictive distribution is correct."""
    q = np.atleast_2d(np.asarray(quantiles, float))
    t = np.asarray(truth, float).ravel()
    lv = np.asarray(levels, float)
    if q.shape[0] != t.shape[0]:
        raise ValueError(f"truth rows {t.shape[0]} != quantile rows {q.shape[0]}")
    out = np.empty(t.shape[0])
    for i in range(t.shape[0]):
        row = q[i]
        ok = np.isfinite(row)
        if ok.sum() < 2:
            out[i] = np.nan
            continue
        # quantile function must be non-decreasing; sort to keep the CDF invertible
        xs, ys = np.sort(row[ok]), lv[ok]
        xi = float(np.interp(t[i], xs, ys, left=ys[0] - (xs[1] - xs[0]),
                             right=ys[-1] + (xs[-1] - xs[-2])))
        out[i] = np.clip(xi, 0.0, 1.0)
    return out


def pit_diagnostics(truth: np.ndarray, quantiles: np.ndarray,
                    levels: list[float]) -> dict:
    pit = pit_values(truth, quantiles, levels)
    pit = pit[np.isfinite(pit)]
    if len(pit) < 5:
        return {"pit_mean": float("nan"), "pit_ks": float("nan"), "n": int(len(pit))}
    ks = float("nan")
    if _sps is not None:
        ks = float(_sps.kstest(pit, "uniform").statistic)
    else:
        s = np.sort(pit)
        i = np.arange(1, len(s) + 1) / len(s)
        ks = float(np.max(np.abs(s - (i - 1.0) / len(s)) + np.abs(s - i / len(s))))
    return {"pit_mean": float(pit.mean()), "pit_ks": ks, "n": int(len(pit)),
            "pit_std": float(pit.std(ddof=1))}


def coverage_by_level(truth: np.ndarray, quantiles: np.ndarray,
                      levels: list[float]) -> dict:
    """Empirical exceedance rate per nominal level, and the average absolute
    calibration gap. A model can win on CRPS and still be useless for VaR."""
    q = np.atleast_2d(np.asarray(quantiles, float))
    t = np.asarray(truth, float).ravel()
    out: dict[str, float] = {}
    gaps = []
    for alpha, lvl in zip(range(q.shape[1]), levels):
        col = q[:, alpha]
        m = np.isfinite(col) & np.isfinite(t)
        if m.sum() < 3:
            continue
        emp = float(np.mean(t[m] < col[m]))
        out[f"coverage_{lvl:.2f}"] = emp
        gaps.append(abs(emp - lvl))
    out["mean_abs_coverage_gap"] = float(np.mean(gaps)) if gaps else float("nan")
    return out


def kupiec_pof(n: int, k: int, alpha: float) -> dict:
    """Kupiec (1995) proportion-of-failures likelihood ratio test on VaR
    exceedances. H0: the exceedance rate equals the nominal rate."""
    if n < 5:
        return {"kupiec_lr": float("nan"), "kupiec_p": float("nan"), "n": n, "k": k}
    p = k / n
    p0 = alpha
    def ll(x: float) -> float:
        return k * math.log(max(1e-12, x)) + (n - k) * math.log(max(1e-12, 1 - x))
    lr = -2.0 * (ll(p0) - ll(p))
    pv = float(_sps.chi2.sf(lr, 1)) if _sps is not None else float(
        math.exp(-lr / 2.0))
    return {"kupiec_lr": float(lr), "kupiec_p": pv, "n": int(n), "k": int(k),
            "hit_rate": p, "alpha": alpha}


def christoffersen_indep(exceed: np.ndarray) -> dict:
    """Christoffersen (2000) independence test on the exceedance indicator:
    clustered VaR breaches are the classic regime-shift failure."""
    e = np.asarray(exceed).astype(int).ravel()
    n = len(e)
    if n < 8:
        return {"christoffersen_lr": float("nan"), "christoffersen_p": float("nan"), "n": n}
    def ll(counts: dict) -> float:
        s = 0.0
        for prev in (0, 1):
            n00, n01 = counts[(prev, 0)], counts[(prev, 1)]
            tot = n00 + n01
            if tot == 0:
                continue
            p1 = n01 / tot
            s += n00 * math.log(max(1e-12, 1 - p1)) + n01 * math.log(max(1e-12, p1))
        return s
    cnt = {(i, j): 0 for i in (0, 1) for j in (0, 1)}
    for a, b in zip(e[:-1], e[1:]):
        cnt[(int(a), int(b))] += 1
    k = int(e.sum())
    p_hat = k / n
    # LR: unrestricted 2-state Markov chain vs i.i.d. exceedances at the same rate
    def ll_iid() -> float:
        return k * math.log(max(1e-12, p_hat)) + (n - k) * math.log(max(1e-12, 1 - p_hat))
    lr_ind = max(0.0, -2.0 * (ll_iid() - ll(cnt)))
    pv = float(_sps.chi2.sf(lr_ind, 1)) if _sps is not None else float(math.exp(-lr_ind / 2))
    return {"christoffersen_lr": float(lr_ind), "christoffersen_p": pv, "n": int(n),
            "k": k, "transitions": {f"{i}{j}": int(v) for (i, j), v in cnt.items()}}


def var_backtest_suite(price: np.ndarray, var_q: np.ndarray, alpha: float,
                       horizon: int = 1) -> dict:
    """Kupiec + Christoffersen + joint LR on a one-sided lower VaR series."""
    y = np.asarray(price, float).ravel()
    v = np.asarray(var_q, float).ravel()
    m = np.isfinite(y) & np.isfinite(v)
    y, v = y[m], v[m]
    exceed = (y < v).astype(int)
    out = kupiec_pof(len(exceed), int(exceed.sum()), alpha)
    out.update(christoffersen_indep(exceed))
    out["joint_lr"] = float(out.get("kupiec_lr", 0.0) + out.get("christoffersen_lr", 0.0))
    out["joint_p"] = (float(_sps.chi2.sf(out["joint_lr"], 2)) if _sps is not None
                      else float(math.exp(-out["joint_lr"] / 2)))
    out["alpha"] = alpha
    out["horizon"] = horizon
    return out


# --------------------------------------------------------------------------- #
# 6. rare-event ranking
# --------------------------------------------------------------------------- #
def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision (PR-AUC). With a 1-3% base rate, ROC-AUC can look good
    while the ranking is useless for flagging events."""
    y = np.asarray(y_true).astype(int).ravel()
    s = np.asarray(y_score, float).ravel()
    m = np.isfinite(s)
    y, s = y[m], s[m]
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    prec = tp / (tp + fp)
    rec = tp / max(1, int(y.sum()))
    # step-wise area under the PR curve
    prev_rec, area = 0.0, 0.0
    for p, r in zip(prec, rec):
        area += p * (r - prev_rec)
        prev_rec = r
    return float(area)


def fbeta(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 2.0) -> float:
    y = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(y_pred).astype(int).ravel()
    tp = float(np.sum((p == 1) & (y == 1)))
    fp = float(np.sum((p == 1) & (y == 0)))
    fn = float(np.sum((p == 0) & (y == 1)))
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    b2 = beta * beta
    return float((1 + b2) * prec * rec / (b2 * prec + rec)) if (b2 * prec + rec) > 0 else 0.0


# --------------------------------------------------------------------------- #
# 7. economic shape
# --------------------------------------------------------------------------- #
def ulcer_index(returns: np.ndarray) -> float:
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return float("nan")
    eq = np.concatenate([[1.0], np.cumprod(1.0 + r)])   # equity path starts at 1
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak - 1.0) * 100.0
    return float(np.sqrt(np.mean(dd**2)))


def max_drawdown_duration(returns: np.ndarray, periods_per_year: int = 252) -> float:
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return float("nan")
    eq = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    peak = np.maximum.accumulate(eq)
    dur = best = 0
    for v, p in zip(eq, peak):
        dur = dur + 1 if v < p else 0
        best = max(best, dur)
    return float(best / periods_per_year)


def market_capture(strat: np.ndarray, bench: np.ndarray) -> dict:
    """Up-market / down-market capture: how much of the benchmark's return a rule
    keeps in rising markets and how much it avoids in falling ones."""
    s = np.asarray(strat, float)
    b = np.asarray(bench, float)
    m = np.isfinite(s) & np.isfinite(b)
    s, b = s[m], b[m]
    up, dn = b > 0, b < 0
    if up.sum() == 0 or dn.sum() == 0:
        return {"capture_up": float("nan"), "capture_down": float("nan")}
    cu = float(s[up].mean() / b[up].mean()) if b[up].mean() != 0 else float("nan")
    cd = float(s[dn].mean() / b[dn].mean()) if b[dn].mean() != 0 else float("nan")
    return {"capture_up": cu, "capture_down": cd,
            "downside_share": float(np.mean(np.where(s[dn] < 0, 1.0, 0.0)))}


def cost_slope(sharpe_by_bps: dict[float, float]) -> dict:
    """Slope of net Sharpe vs transaction cost (bps) and the break-even cost.
    A slope of -0.1/10bps and a 40bps break-even is a different animal from a
    strategy that dies at 5bps."""
    pts = sorted(((float(k), float(v)) for k, v in sharpe_by_bps.items()
                  if np.isfinite(v)))
    if len(pts) < 2:
        return {"cost_slope_per_10bps": float("nan"), "break_even_bps": float("nan")}
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    A = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope, inter = float(beta[1]), float(beta[0])
    be = float(-inter / slope) if slope < 0 else float("inf")
    return {"cost_slope_per_10bps": slope * 10.0, "break_even_bps": be,
             "gross_intercept": inter}


# --------------------------------------------------------------------------- #
# scaled accuracy
# --------------------------------------------------------------------------- #
def relative_skill(mase_model: float, mase_naive: float) -> float:
    """RS = 1 - MASE_model/MASE_naive: scale-free skill vs the naive rule, so a
    table across assets/frequencies is readable without footnotes."""
    if not np.isfinite(mase_model) or not np.isfinite(mase_naive) or mase_naive == 0:
        return float("nan")
    return float(1.0 - mase_model / mase_naive)


def scaled_crps(crps: float, scale: float) -> float:
    """sCRPS = CRPS / (MAD of the training series): comparable across assets whose
    price levels differ by orders of magnitude."""
    if not np.isfinite(crps) or not np.isfinite(scale) or scale == 0:
        return float("nan")
    return float(crps / scale)
