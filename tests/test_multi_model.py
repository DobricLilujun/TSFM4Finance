"""Contract tests for the multi-model sweep: store, protocol, trainers, metrics.

These run in the core env and touch no GPU model weights (the model adapters are
covered by ``--stage status`` / reports/PROGRESS.md instead).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tsfm4finance.bench import finance_metrics as FM
from tsfm4finance.bench.config import load_grid
from tsfm4finance.bench.finance_metrics import (benjamini_hochberg, cost_slope,
                                                deflated_sharpe_ratio, diebold_mariano,
                                                kupiec_pof, pit_diagnostics, qlike,
                                                relative_skill, stationary_bootstrap_spa,
                                                ulcer_index)
from tsfm4finance.bench.protocol import HORIZONS, build_windows
from tsfm4finance.bench.state import CellStore, cell_id
from tsfm4finance.bench.trading import positions, signal_from_forecast, trading_returns


# ------------------------------------------------------------------ state store
def test_cell_id_is_stable_and_unique():
    a = cell_id({"stage": "forecast", "series": "x", "model": "m", "adaptation": "zero",
                 "horizon": 21, "span": "auto", "seed": 0})
    b = cell_id({"stage": "forecast", "series": "x", "model": "m", "adaptation": "zero",
                 "horizon": 21, "span": "auto", "seed": 0})
    c = cell_id({"stage": "forecast", "series": "x", "model": "m", "adaptation": "fs16",
                 "horizon": 21, "span": "auto", "seed": 0})
    assert a == b and a != c


def test_store_resume_after_crash(tmp_path):
    st = CellStore(tmp_path / "s.sqlite3")
    specs = [{"stage": "forecast", "series": f"s{i}", "model": "m", "adaptation": "zero",
              "horizon": 5, "span": "auto", "seed": 0} for i in range(5)]
    todo, _, _ = st.plan(specs)
    assert len(todo) == 5
    st.claim(todo[0]["cell_id"])                       # left in `running` by a crash
    st.finish(todo[1]["cell_id"], {"aggregate": {"mase": 1.0}})
    st.fail(todo[2]["cell_id"], "boom")
    st.skip(todo[3]["cell_id"], "no-quantiles")
    st2 = CellStore(tmp_path / "s.sqlite3")            # reopen: crash-safe
    assert st2.reset_running() == 1                    # abandoned cell re-queued
    assert st2.retry_failed() == 1
    c = st2.counts()["forecast"]
    assert c["done"] == 1 and c["skipped"] == 1 and c["pending"] == 3


def test_plan_is_idempotent(tmp_path):
    st = CellStore(tmp_path / "s2.sqlite3")
    specs = [{"stage": "forecast", "series": "s", "model": "m", "adaptation": "zero",
              "horizon": 5, "span": "auto", "seed": 0}]
    todo1, _, _ = st.plan(specs)
    assert len(todo1) == 1
    st.finish(todo1[0]["cell_id"], {"aggregate": {"mase": 1.0}})
    assert st.plan(specs)[1] == 1                      # counted as already done
    assert len(st.plan(specs)[0]) == 0                 # not re-queued
    assert len(st.plan(specs, force=True)[0]) == 1     # --force re-queues it


# --------------------------------------------------------------------- protocol
def test_windows_never_leak_and_are_disjoint():
    import pandas as pd
    n = 600
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    y = np.cumsum(np.random.RandomState(0).normal(0, 1, n)) + 100
    df = pd.DataFrame({"close": y}, index=idx)
    tr, va, te = df.iloc[:400], df.iloc[400:500], df.iloc[500:]
    for h in HORIZONS:
        wins, info = build_windows(tr, va, te, horizon=h, span="auto")
        assert info["no_leakage"] is True
        assert info["coverage"] > 0.5
        starts = [w.origin for w in wins]
        assert all(b - a >= h for a, b in zip(starts, starts[1:]))   # stride >= h
        for w in wins:
            assert len(w.truth) == h
            assert w.context[-1] == df["close"].iloc[w.origin - 1]   # ends before origin


# ----------------------------------------------------------------- finance metrics
def test_dm_prefers_the_better_forecast():
    rng = np.random.RandomState(1)
    y = rng.normal(size=400)
    bad = y + rng.normal(0, 2.0, 400)
    good = y + rng.normal(0, 0.6, 400)
    out = diebold_mariano(bad, good, horizon=5)
    # positive t with A = bad => A's losses are significantly larger
    assert out["dm_t"] > 0 and out["p_value"] < 0.05 and out["favours"] == "B"
    assert out["mz_t"] > 0          # MSE-adjusted variant agrees


def test_dsr_penalises_trial_count():
    rng = np.random.RandomState(2)
    r = rng.normal(0.002, 0.01, 500)
    one = deflated_sharpe_ratio(r, n_trials=1)
    many = deflated_sharpe_ratio(r, n_trials=1000)
    assert one["dsr"] >= many["dsr"]          # more trials -> less credible
    assert many["p_value"] >= one["p_value"]


def test_spa_rejects_pure_noise():
    rng = np.random.RandomState(3)
    bench = rng.normal(0, 0.01, 300)
    null = bench + rng.normal(0, 0.002, 300)          # no edge: mean diff ~ 0
    out = stationary_bootstrap_spa(null, bench, n_boot=400, seed=4)
    assert out["spa_p"] > 0.05
    edge = bench + rng.normal(0.002, 0.002, 300)      # real edge
    out2 = stationary_bootstrap_spa(edge, bench, n_boot=400, seed=5)
    assert out2["spa_p"] < 0.05 and out2["max_t"] > out2["crit95"]
    multi = stationary_bootstrap_spa(np.stack([null, edge], axis=1), bench, n_boot=400, seed=6)
    assert multi["n_strategies"] == 2 and multi["spa_p"] < 0.10


def test_bh_fdr_finds_the_signal():
    p = np.array([0.001, 0.004, 0.02, 0.6, 0.9])
    out = benjamini_hochberg(p, q=0.10)
    assert 0 in out["rejected"] and 1 in out["rejected"]
    assert out["p_adj"][3] > 0.10


def test_calibration_and_var_tests():
    rng = np.random.RandomState(5)
    y = rng.normal(size=300)
    q = np.stack([np.full(300, -1.645), np.zeros(300), np.full(300, 1.645)], axis=1)
    d = pit_diagnostics(y, q, levels=[0.05, 0.5, 0.95])
    assert abs(d["pit_mean"] - 0.5) < 0.15 and 0 <= d["pit_ks"] <= 1
    k = kupiec_pof(n=300, k=15, alpha=0.05)
    assert k["n"] == 300 and 0 <= k["kupiec_p"] <= 1 and k["hit_rate"] == pytest.approx(0.05)
    ch = FM.christoffersen_indep((y < -1.645).astype(int))
    assert 0 <= ch["christoffersen_p"] <= 1


def test_qlike_and_vol_ratio():
    v = np.abs(np.random.RandomState(6).normal(1, 0.2, 200)) + 0.1
    # QLIKE is minimised by the true variance: a mis-scaled forecast pays for it
    assert qlike(v, v) < qlike(v * 1.05, v) < qlike(v * 2, v)
    assert np.isfinite(FM.vol_zscore_ratio(v * 1.2, v))


def test_economic_extras_shape():
    rng = np.random.RandomState(7)
    r = rng.normal(0.0005, 0.01, 500)
    assert ulcer_index(r) >= 0
    cs = cost_slope({0.0: 1.2, 10.0: 0.8, 30.0: 0.2})
    assert cs["cost_slope_per_10bps"] < 0 and cs["break_even_bps"] > 10
    assert relative_skill(mase_model=0.9, mase_naive=1.0) == pytest.approx(0.1)


# ------------------------------------------------------------------- trading layer
def test_costs_are_charged_on_turnover_only():
    pos = np.array([1.0, 1.0, 1.0])
    fr = np.zeros(3)
    assert trading_returns(pos, fr, cost_bps=10.0)[1:].sum() == 0     # no turnover
    pos2 = np.array([1.0, -1.0])
    assert trading_returns(pos2, np.zeros(2), cost_bps=10.0)[1] == pytest.approx(-2e-3)


def test_signal_from_forecast_uses_the_band():
    ctx = np.linspace(100, 110, 64)
    s, sig = signal_from_forecast(ctx, np.array([112.0, 115.0]), None, 2)
    assert s > 0 and sig > 0
    w = positions(np.array([0.02, -0.05]), np.array([0.01, 0.02]), "voltarget")
    assert np.all(np.abs(w) <= 1.0)


# ----------------------------------------------------------------------- registry
def test_grid_registry_is_consistent():
    g = load_grid()
    ms = g.models()
    assert len(ms) >= 20
    for k, m in ms.items():
        assert m.env in g.envs, f"{k} points at an unknown env {m.env}"
        assert m.family and m.tier in ("A", "B", "C")
    # every advertised capability key is one the sweep understands
    known = {"forecast", "quantiles", "embed", "clf_head", "anomaly_head", "finetune",
             "multivariate", "covariates", "sample_paths", "panel_in", "ohlcv_out"}
    for k, m in ms.items():
        assert set(m.caps) <= known, (k, set(m.caps) - known)


def test_horizon_switch_matches_paper_spec():
    import pandas as pd
    df = pd.DataFrame({"close": np.arange(1.0, 801.0)},
                      index=pd.date_range("2010-01-01", periods=800, freq="B"))
    tr, va, te = df.iloc[:500], df.iloc[500:650], df.iloc[650:]
    short, i1 = build_windows(tr, va, te, horizon=5, span="auto")
    longw, i2 = build_windows(tr, va, te, horizon=63, span="auto")
    assert i1["span"] == "test" and i2["span"] == "val+test"
