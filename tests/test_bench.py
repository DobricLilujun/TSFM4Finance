"""Tests for the spec §4.1 benchmark protocol, task datasets and trading metrics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsfm4finance.paths import DATA
from tsfm4finance.core.metrics import (crps_from_quantiles, trading_metrics, max_drawdown,
                                       normalized_difference, QUANTILE_LEVELS)
from tsfm4finance.bench.protocol import build_windows, NaiveEngine, SeasonalNaiveEngine

DS = "sp500_daily"


def _splits(name: str = DS):
    d = DATA / "open" / name
    return (pd.read_parquet(d / "train.parquet"),
            pd.read_parquet(d / "validation.parquet"),
            pd.read_parquet(d / "test.parquet"))


def test_windows_are_non_overlapping_and_in_stride_order():
    tr, va, te = _splits()
    W, info = build_windows(tr, va, te, horizon=5)
    origins = [w.origin for w in W]
    assert origins == sorted(origins)
    assert np.all(np.diff(origins) == 5)                    # stride == horizon
    for a, b in zip(W, W[1:]):
        assert a.origin + len(a.truth) <= b.origin          # no overlap


def test_no_leakage_context_always_precedes_truth():
    tr, va, te = _splits()
    for h in (1, 5, 21, 63):
        W, info = build_windows(tr, va, te, horizon=h)
        assert info["no_leakage"], info
        n_train = len(tr)
        for w in W:
            assert len(w.context) + len(w.truth) <= w.origin + len(w.truth)
            assert w.origin >= n_train                      # eval span is held out
            assert len(w.context) <= 512


def test_long_horizons_switch_to_val_plus_test_span():
    tr, va, te = _splits()
    _, i_short = build_windows(tr, va, te, horizon=5)
    W_long, i_long = build_windows(tr, va, te, horizon=63)
    assert i_short["span"] == "test"
    assert i_long["span"] == "val+test"
    assert W_long[0].origin >= len(tr)                      # never starts inside train
    assert i_long["n_origins"] >= 3


def test_context_capped_at_max_context():
    tr, va, te = _splits()
    W, info = build_windows(tr, va, te, horizon=5, max_context=128)
    assert all(len(w.context) <= 128 for w in W)
    assert info["max_context"] == 128


def test_naive_engine_equals_persistence():
    ctx = [np.arange(100.0), np.linspace(0, 5, 60)]
    pts, q = NaiveEngine().predict_batch(ctx, 7)
    assert pts.shape == (2, 7)
    assert np.allclose(pts[0], 99.0)
    assert q is None


def test_seasonal_naive_repeats_last_season():
    ctx = [np.arange(30.0)]
    pts, _ = SeasonalNaiveEngine(season=5).predict_batch(ctx, 7)
    assert list(pts[0]) == [25, 26, 27, 28, 29, 25, 26]


def test_trading_zero_cost_net_equals_gross_and_costs_only_reduce():
    w = np.array([1.0, -1.0, 1.0, 1.0])
    r = np.log(np.array([1.01, 0.99, 1.02, 1.01]))
    a = trading_metrics(w, r, cost_bps=0.0)
    b = trading_metrics(w, r, cost_bps=10.0)
    c = trading_metrics(w, r, cost_bps=30.0)
    assert abs(a["ann_return_net"] - a["ann_return_gross"]) < 1e-12
    assert a["ann_return_net"] > b["ann_return_net"] > c["ann_return_net"]
    assert a["sharpe_net"] >= b["sharpe_net"] >= c["sharpe_net"]


def test_trading_buy_and_hold_recovers_asset_return():
    r = np.log(np.array([1.01, 1.02, 0.99, 1.005, 1.01]))
    m = trading_metrics(np.ones(len(r)), r, cost_bps=0.0)
    assert abs(m["cum_return_net"] - (np.exp(r.sum()) - 1)) < 1e-9
    assert m["hit_rate"] == 0.8          # one of the five periods is negative


def test_max_drawdown_known_case():
    # +25%, -40%, +10%  -> wealth 1.25, 0.75, 0.825 -> dd = 1 - 0.75/1.25 = 0.40
    dd = max_drawdown(np.array([0.25, -0.40, 0.10]))
    assert abs(dd - 0.40) < 1e-9


def test_crps_lower_for_sharper_correct_forecast():
    y = np.array([1.0, 2.0, 3.0])
    tight = np.vstack([y + d for d in (-0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2)])
    wide = np.vstack([y + d for d in (-3, -2.25, -1.5, -0.75, 0, 0.75, 1.5, 2.25, 3)])
    assert crps_from_quantiles(y, tight) < crps_from_quantiles(y, wide)


def test_normalized_difference_one_equals_seasonal_naive():
    hist = np.cumsum(np.random.RandomState(0).randn(80)) + 100
    truth = hist[-10:]
    naive = hist[-11:-1]
    nd = normalized_difference(naive, truth, history=hist)
    assert abs(nd - 1.0) < 1e-9


# ---------------- task datasets ----------------#
@pytest.mark.parametrize("suffix,task", [("dir5", "classify"), ("volreg", "classify"),
                                         ("extreme", "anomaly"), ("trade", "trading")])
def test_real_task_datasets_exist_and_are_spec_compliant(suffix, task):
    d = DATA / "open" / f"{DS}_{suffix}"
    assert d.exists(), f"missing dataset {d}"
    meta = json.loads((d / "meta.json").read_text())
    assert meta["task"] == task and meta["open"] is True
    tr = pd.read_parquet(d / "train.parquet")
    te = pd.read_parquet(d / "test.parquet")
    assert len(tr) and len(te)
    if suffix in ("dir5", "volreg", "extreme"):
        assert set(np.unique(tr["label"].dropna().to_numpy())) <= (
            {0.0, 1.0} if suffix != "volreg" else {0.0, 1.0, 2.0})


def test_direction_labels_use_the_future_and_have_no_leakage_into_features():
    """label[t] must equal sign of the return over (t, t+5]; prices at t are the features."""
    d = DATA / "open" / f"{DS}_dir5"
    full = pd.concat([pd.read_parquet(d / f"{s}.parquet") for s in
                      ("train", "validation", "test")], ignore_index=True)
    c = full["close"].to_numpy(dtype=float)
    lab = full["label"].to_numpy(dtype=float)
    t = 100
    expected = float(np.log(c[t + 5] / c[t]) > 0)
    assert lab[t] == expected


def test_extreme_labels_use_only_prior_data():
    """The expanding-MAD threshold at t must come from returns strictly before t."""
    d = DATA / "open" / f"{DS}_extreme"
    full = pd.concat([pd.read_parquet(d / f"{s}.parquet") for s in
                      ("train", "validation", "test")], ignore_index=True)
    c = full["close"].to_numpy(dtype=float)
    lab = full["label"].to_numpy(dtype=float)
    # ret[t] = log return AT t ; the threshold at t may only use ret[<t]
    ret = np.r_[0.0, np.diff(np.log(c))]
    rec = np.zeros(len(c))
    for t in range(1, len(c)):
        hist = ret[:t]
        if len(hist) < 30:
            continue
        med = np.median(hist)
        mad = np.median(np.abs(hist - med))
        if mad > 0 and abs(ret[t] - med) > 3.0 * 1.4826 * mad:
            rec[t] = 1.0
    assert np.array_equal(rec, lab)
    # the check has power: a threshold fitted on FUTURE returns gives different labels,
    # so equality above really does pin the no-look-ahead definition
    leaky = np.zeros(len(c))
    for t in range(1, len(c) - 30):
        hist = ret[t + 1:t + 31]
        med = np.median(hist)
        mad = np.median(np.abs(hist - med))
        if mad > 0 and abs(ret[t] - med) > 3.0 * 1.4826 * mad:
            leaky[t] = 1.0
    assert not np.array_equal(leaky, lab)
    assert np.all(lab[:30] == 0.0)


def test_volreg_cutpoints_come_from_train_only():
    d = DATA / "open" / f"{DS}_volreg"
    meta = json.loads((d / "meta.json").read_text())
    tr = pd.read_parquet(d / "train.parquet")
    counts = tr["label"].value_counts()
    # terciles fitted on train => train classes are near-equal in size
    assert len(counts) == 3
    assert counts.min() / counts.max() > 0.9
    assert "TRAIN split" in meta["notes"]


# --------------------------------------------------------------------------- #
# completion invariants for the executed benchmark grid
# --------------------------------------------------------------------------- #
import json as _json
from tsfm4finance.bench.run import RESULTS_PATH, LEADERBOARD, HORIZONS, FT_VARIANTS
from tsfm4finance.bench.tasks import REAL_SERIES

RESULTS = _json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
GENERIC = ["naive", "seasonal_naive", "moving_average", "arima", "chronos",
           "timesfm@128", "timesfm@256", "timesfm@512"]


def test_forecast_grid_is_complete_and_error_free():
    want = len(REAL_SERIES) * len(HORIZONS) * (len(GENERIC) + len(FT_VARIANTS))
    assert len(RESULTS["forecast"]) == want == 660
    assert not [k for k, v in RESULTS["forecast"].items() if "error" in v]
    for base, _d in REAL_SERIES:
        for h in HORIZONS:
            for m in GENERIC:
                assert f"{base}:h{h}:{m}" in RESULTS["forecast"]
            for tag, _K in FT_VARIANTS:
                assert f"{base}:h{h}:timesfm_ft:{tag}:{base}" in RESULTS["forecast"]


def test_every_forecast_row_asserts_no_leakage_and_covers_the_span():
    for key, val in RESULTS["forecast"].items():
        info = val.get("info")
        assert info and info["no_leakage"] is True, key
        assert info["coverage"] > 0.5, key
        assert info["stride"] == info["horizon"], key          # non-overlapping windows
        assert info["max_context"] == 512, key
        assert val["aggregate"]["n_origins"] == info["n_origins"]


def test_finetune_ladder_complete_for_every_series():
    assert len(RESULTS["finetune"]) == len(REAL_SERIES) * len(FT_VARIANTS) == 36
    for base, _d in REAL_SERIES:
        for tag, K in FT_VARIANTS:
            e = RESULTS["finetune"][f"{base}:{tag}"]
            assert "error" not in e
            assert e["k_windows"] == (K if K else e["k_windows"])
            assert e["steps"] > 0 and e["seconds"] > 0
            assert e["final_loss"] < e["first_loss"]          # the run actually optimized


def test_trading_entries_cover_both_rules_three_costs_and_ft_signals():
    for base, _d in REAL_SERIES:
        e = RESULTS["trading"][f"{base}_trade"]
        for node in ("model", "model_fs16", "model_fs64", "model_full"):
            for mode in ("sign", "voltarget"):
                block = e[node][mode]
                assert set(block) >= {"0bps", "10bps", "30bps", "cost_sensitivity_sharpe"}
                for cb in ("0bps", "10bps", "30bps"):
                    m = block[cb]
                    assert m["sharpe_net"] <= m["sharpe_gross"] + 1e-9   # costs never help
                    assert m["turnover_total"] >= 0.0
                    assert 0.0 <= m["hit_rate"] <= 1.0
        assert "buy_and_hold" in e["benchmarks"] and "momentum" in e["benchmarks"]
        assert e["n_trades"] >= 10 and e["horizon_days"] == 5


def test_anomaly_reports_oracle_reference_and_fair_competitors():
    for base, _d in REAL_SERIES:
        e = RESULTS["anomaly"][f"{base}_extreme"]
        assert {"oracle_label_rule", "stat_zscore", "timesfm_bandviolation"} <= set(e)
        assert "n_pos_eval" in e and "n_pos_fit" in e


def test_leaderboard_is_strict_json_and_tagged_by_protocol():
    raw = Path(LEADERBOARD).read_text()
    recs = _json.loads(raw)                                  # strict JSON (no bare nan)
    assert "nan" not in raw.replace("nanos", "").replace("financial", "")
    spec = [r for r in recs if r.get("protocol") == "spec-4.1-rolling"]
    assert len(spec) == 132                                  # 12 series x 11 models @ h=21
    assert all(isinstance(r["metrics"], dict) for r in spec)
    assert all(r.get("horizon") == 21 for r in spec)
    assert sum(1 for r in recs if r.get("protocol") == "legacy-single-window") > 0


def test_report_markdown_contains_every_section():
    md = (RESULTS_PATH.parent / "02_timesfm_benchmark.md").read_text()
    for head in ("## 1. Forecasting", "## 2. Full fine-tuning", "## 3. Classification",
                 "## 4. Extreme-move anomaly", "## 5. Quant-trading task",
                 "## 6. Protocol checks"):
        assert head in md, head
    assert "NDp" in md and "cost sensitivity" in md.lower()


def test_finetune_contract_is_full_parameter_not_head_only():
    src = (RESULTS_PATH.parents[1] / "tsfm4finance" / "bench" / "finetune.py").read_text()
    assert "prm.requires_grad_(True)" in src                 # every parameter trainable
    assert "AdamW(module.parameters()" in src                # optimizer sees all of them
    assert "requires_grad_(False)" not in src                # nothing is frozen
