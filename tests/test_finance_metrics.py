"""Numeric tests for the finance-specific metrics.

Reference values are worked out by hand (or against a closed form) rather than
snapshot-generated, so a wrong implementation fails instead of being frozen in.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tsfm4finance.bench.finance_metrics import (
    benjamini_hochberg, christoffersen_indep, cost_slope,
    deflated_sharpe_ratio, diebold_mariano, hac_variance, kupiec_pof,
    market_capture, max_drawdown_duration, mz_variance_regression, pit_diagnostics,
    pit_values, coverage_by_level, pr_auc, qlike, relative_skill,
    scaled_crps, stationary_bootstrap_spa, ulcer_index, var_backtest_suite,
    vol_zscore_ratio,
)


# ---------------------------------------------------------------- paired tests
def test_hac_variance_lags_zero_is_plain_variance_of_mean():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # g0 = mean squared deviation = 2.0, var(mean) = g0/n = 0.4
    assert hac_variance(x, 0) == pytest.approx(2.0)
    assert hac_variance(x, 0) / len(x) == pytest.approx(0.4)


def test_hac_variance_handles_positive_autocorrelation():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert hac_variance(x, 2) > hac_variance(x, 0)      # positively ordered series


def test_dm_prefers_the_better_model_and_is_zero_when_identical():
    rng = np.random.default_rng(7)
    truth = rng.normal(size=200).cumsum()
    good = truth + rng.normal(scale=0.3, size=200)
    bad = truth + rng.normal(scale=2.0, size=200)
    out = diebold_mariano(bad - truth, good - truth, horizon=1)
    assert out["dm_t"] > 0 and out["favours"] == "B"
    assert out["p_value"] < 1e-6
    same = diebold_mariano(good - truth, good - truth, horizon=1)
    assert same["dm_t"] == pytest.approx(0.0)


def test_dm_mz_correction_shrinks_statistic():
    rng = np.random.default_rng(11)
    e_a, e_b = rng.normal(scale=1.0, size=120), rng.normal(scale=1.2, size=120)
    out = diebold_mariano(e_a, e_b, horizon=21)
    assert abs(out["mz_t"]) < abs(out["dm_t"])
    assert out["hac_lags"] == 20


def test_dm_is_inconclusive_on_too_few_windows():
    out = diebold_mariano(np.zeros(3), np.ones(3), horizon=5)
    assert math.isnan(out["dm_t"]) and out["favours"] == "inconclusive"


# ---------------------------------------------------------------- vol metrics
def test_qlike_closed_form_when_forecast_equals_outcome():
    v = np.array([0.01, 0.04, 0.09])
    assert qlike(v, v) == pytest.approx(float(np.mean(np.log(v)) + 1.0))


def test_qlike_penalises_zero_variance_forecasts_by_ignoring_them():
    v = np.array([0.04, 0.04])
    assert math.isnan(qlike(np.array([0.0, 0.0]), v))


def test_mz_variance_regression_recovers_exact_affine_map():
    x = np.linspace(0.01, 0.1, 40)
    y = 2.0 * x
    out = mz_variance_regression(x, y)
    assert out["slope"] == pytest.approx(2.0, abs=1e-8)
    assert out["intercept"] == pytest.approx(0.0, abs=1e-8)
    assert out["r2"] == pytest.approx(1.0, abs=1e-8)


def test_vol_zscore_sign():
    assert vol_zscore_ratio(np.array([0.4, 0.4, 0.4]), np.array([0.2, 0.2, 0.2])) < 0
    assert vol_zscore_ratio(np.array([0.2, 0.2, 0.2]), np.array([0.4, 0.4, 0.4])) > 0


# ---------------------------------------------------------------- VaR backtests
def test_kupiec_passes_at_the_nominal_rate():
    out = kupiec_pof(250, 25, 0.10)
    assert out["kupiec_lr"] == pytest.approx(0.0, abs=1e-9)
    assert out["kupiec_p"] == pytest.approx(1.0, abs=1e-6)


def test_kupiec_rejects_a_wrong_hit_rate():
    out = kupiec_pof(250, 60, 0.10)
    assert out["kupiec_p"] < 1e-6


def test_christoffersen_flags_clustered_breaches():
    clustered = np.array([0] * 100 + [1] * 20 + [0] * 100 + [1] * 20, dtype=int)
    alternating = np.tile([0, 1], 62)[:124]
    c1, c2 = christoffersen_indep(clustered), christoffersen_indep(alternating)
    assert c1["christoffersen_p"] < 0.05
    assert c2["christoffersen_lr"] > 0.0


def test_var_backtest_suite_reports_joint_chi2():
    rng = np.random.default_rng(3)
    y = rng.normal(size=500)
    var_q = np.full(500, -1.2816)          # ~10% of a standard normal below this
    out = var_backtest_suite(y, var_q, 0.10)
    assert abs(out["hit_rate"] - 0.10) < 0.05
    assert out["kupiec_p"] > 0.05
    assert out["joint_p"] <= 1.0


# ---------------------------------------------------------------- calibration
def test_pit_is_half_when_truth_sits_at_the_predicted_median():
    q = np.array([[0.0, 1.0, 2.0]] * 5)                      # levels .1 .5 .9
    truth = np.array([1.0] * 5)
    pit = pit_values(truth, q, [0.1, 0.5, 0.9])
    assert np.allclose(pit, 0.5)
    d = pit_diagnostics(truth, q, [0.1, 0.5, 0.9])
    assert d["pit_mean"] == pytest.approx(0.5)


def test_coverage_matches_nominal_levels_on_correct_quantiles():
    rng = np.random.default_rng(5)
    draws = rng.normal(size=4000)
    lv = [0.05, 0.1, 0.5, 0.9, 0.95]
    z = np.array([-1.6449, -1.2816, 0.0, 1.2816, 1.6449])
    q = np.repeat(z[None, :], 4000, axis=0) + draws[:, None] * 0  # exact normal q
    out = coverage_by_level(draws, q, lv)
    for lvl in lv:
        assert abs(out[f"coverage_{lvl:.2f}"] - lvl) < 0.02
    assert out["mean_abs_coverage_gap"] < 0.02


# ---------------------------------------------------------------- rare events
def test_pr_auc_perfect_and_reversed_rankings():
    y = np.array([0, 1, 0, 1])
    assert pr_auc(y, np.array([0.1, 0.9, 0.2, 0.8])) == pytest.approx(1.0)
    # reversed: AP = (1/3)(1/2) + (1/2)(1/2) = 5/12
    assert pr_auc(y, np.array([0.9, 0.1, 0.8, 0.2])) == pytest.approx(5 / 12)
    assert math.isnan(pr_auc(np.zeros(5, dtype=int), np.linspace(0, 1, 5)))


def test_fbeta_weights_recall():
    from tsfm4finance.bench.finance_metrics import fbeta
    y = np.array([1, 1, 1])
    p = np.array([1, 1, 0])
    # precision 1, recall 2/3 -> F2 = 5PR/(4P+R) = 10/14
    assert fbeta(y, p, beta=2.0) == pytest.approx(10.0 / 14.0)
    assert fbeta(y, p, beta=1.0) == pytest.approx(0.8)   # F1 = 2PR/(P+R)


# ---------------------------------------------------------------- Sharpe family
def test_dsr_null_grows_with_the_number_of_trials():
    rng = np.random.default_rng(9)
    r = rng.normal(loc=0.001, scale=0.01, size=400)
    a = deflated_sharpe_ratio(r, n_trials=1)
    b = deflated_sharpe_ratio(r, n_trials=500)
    assert a["sr0"] == pytest.approx(0.0)
    assert b["sr0"] > 0.5
    assert b["dsr"] < a["dsr"]
    assert 0.0 <= b["dsr"] <= 1.0 and 0.0 < a["dsr"] < 1.0


def test_spa_detects_a_genuinely_better_strategy_and_rejects_a_bad_one():
    rng = np.random.default_rng(13)
    bench = rng.normal(size=300)
    good = np.vstack([rng.normal(size=300) for _ in range(9)]
                     + [bench + rng.normal(loc=0.5, size=300)])
    bad = -np.abs(np.vstack([bench + rng.normal(size=300) for _ in range(10)]))
    assert stationary_bootstrap_spa(good, bench, n_boot=600)["spa_p"] < 0.05
    assert stationary_bootstrap_spa(bad, bench, n_boot=600)["spa_p"] > 0.9


def test_benjamini_hochberg_step_up():
    out = benjamini_hochberg([0.001, 0.01, 0.04, 0.6], q=0.10)
    assert sorted(out["rejected"]) == [0, 1, 2]
    assert out["p_adj"][0] == pytest.approx(0.004)
    assert out["p_adj"][3] == pytest.approx(0.6)


def test_cost_slope_and_break_even():
    # Sharpe(bps) = 1 - 0.002*bps  ->  slope -0.02 per 10bps, break-even 500bps
    d = {0.0: 1.0, 100.0: 0.8, 300.0: 0.4}
    out = cost_slope(d)
    assert out["cost_slope_per_10bps"] == pytest.approx(-0.02, abs=1e-6)
    assert out["break_even_bps"] == pytest.approx(500.0, abs=1e-6)


def test_ulcer_and_drawdown_duration_reference_values():
    r = np.array([-0.5, 0.0, 0.0, 1.0])   # equity 1 -> .5 -> .5 -> .5 -> 1
    assert ulcer_index(r) == pytest.approx(math.sqrt(1500.0))
    assert max_drawdown_duration(r, periods_per_year=4) == pytest.approx(0.75)


def test_market_capture_identity():
    b = np.array([0.02, -0.03, 0.01, -0.01])
    out = market_capture(b, b)
    assert out["capture_up"] == pytest.approx(1.0)
    assert out["capture_down"] == pytest.approx(1.0)


def test_relative_skill_and_scrps():
    assert relative_skill(4.0, 5.0) == pytest.approx(0.2)
    assert relative_skill(6.0, 5.0) == pytest.approx(-0.2)
    assert scaled_crps(0.5, 2.0) == pytest.approx(0.25)
    assert math.isnan(relative_skill(float("nan"), 5.0))
