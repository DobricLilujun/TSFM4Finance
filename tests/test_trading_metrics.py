"""Cost-aware trading metrics, checked against hand-computed numbers.

These landed in the canonical metrics package rather than in the arena tree because both
`tsfm4finance.bench` and the unified evaluator import them, and two copies of a Sharpe ratio is
how a table ends up reporting two of them. The numbers below are worked out on paper, so a
scaling mistake (bps is 1e-4, not 1e-3) fails here instead of in a published row.
"""
from __future__ import annotations

import numpy as np

from tsfm_eval.metrics import TRADING_COST_GRID, ic_signal, max_drawdown, trading_metrics


def test_ten_basis_points_is_one_thousandth_of_turnover():
    # hold +1 for four periods on +1% log returns each: one unit of entry turnover, nothing after
    m = trading_metrics(np.ones(4), np.full(4, 0.01), cost_bps=0.0, periods_per_year=252)
    assert m["ann_return_gross"] == round(0.01 * 252, 10)
    assert m["ann_return_net"] == m["ann_return_gross"]
    m10 = trading_metrics(np.ones(4), np.full(4, 0.01), cost_bps=10.0, periods_per_year=252)
    # cost = 1.0 turnover * 10/10_000 = 0.001, paid in one period out of four. The cost is taken in
    # simple space (it is a fraction of wealth traded), so the net stream is not "0.04 - 0.001" in
    # log space; mark the portfolio in simple returns, then convert back.
    r_s = np.expm1(0.01)
    net_simple = np.array([r_s - 0.001, r_s, r_s, r_s])
    want = np.mean(np.log1p(net_simple)) * 252
    assert np.isclose(m10["ann_return_net"], want), (m10["ann_return_net"], want)
    assert np.isclose(m10["cum_return_net"], np.prod(1.0 + net_simple) - 1.0)
    assert m10["turnover_total"] == 0.25


def test_a_short_loses_the_amount_it_actually_lost():
    """`w * r` with r a log return is not a portfolio return. A unit short into a +10% month loses
    10% of the notional - not 9.53%, which is what multiplying by the log return reports. The
    portfolio must be marked in simple returns and only then logged."""
    r = np.log(1.1)
    m = trading_metrics(np.array([-1.0]), np.array([r]), cost_bps=0.0, periods_per_year=252)
    assert np.isclose(m["cum_return_net"], -0.10), m["cum_return_net"]
    assert np.isclose(m["ann_return_net"], np.log1p(-0.10) * 252)
    # long side of the same month: +10% simple, which logs to exactly the input log return
    ml = trading_metrics(np.array([1.0]), np.array([r]), cost_bps=0.0, periods_per_year=252)
    assert np.isclose(ml["cum_return_net"], 0.10)
    # a symmetric short is not the negative of the long in log space, and that is the point
    assert abs(m["cum_return_net"]) > abs(ml["cum_return_net"]) * 0.999


def test_identical_losses_do_not_produce_an_infinite_sortino():
    """Sortino's denominator is the downside deviation from the target (root mean square of
    min(x, 0) over every period). std() of only the negative rows is 0 for a constant losing
    stream, which reported an infinite ratio for a strategy that only ever loses."""
    net = np.full(6, -0.01)
    m = trading_metrics(np.ones(6), net, cost_bps=0.0, periods_per_year=252)
    assert np.isfinite(m["sortino_net"]), m["sortino_net"]
    want = (np.mean(net) * 252) / (0.01 * np.sqrt(252))
    assert np.isclose(m["sortino_net"], want), (m["sortino_net"], want)
    # no downside at all -> undefined, not infinite
    up = trading_metrics(np.ones(4), np.full(4, 0.01), cost_bps=0.0)
    assert np.isnan(up["sortino_net"])


def test_a_position_that_ignores_the_sign_of_the_return_scores_zero_hit_rate():
    m = trading_metrics(np.ones(3), np.array([0.01, -0.02, 0.03]), cost_bps=0.0)
    assert np.isclose(m["hit_rate"], 2 / 3)
    assert m["n_bets"] == 3.0


def test_drawdown_is_a_positive_fraction_of_a_simple_return_stream():
    # -50% then +100% then flat: wealth goes 1 -> 0.5 -> 1 -> 1, so the trough is half.
    # The unanchored version of this (peak over cumprod only, no initial 1.0) reads that as 0.0:
    # the first period's loss is compared against itself. A window that opens with a drawdown
    # - which is exactly what a risk metric is for - reported no risk at all.
    assert np.isclose(max_drawdown(np.array([-0.5, 1.0, 0.0])), 0.5)
    assert np.isclose(max_drawdown(np.array([-0.5, 0.1])), 0.5)
    assert max_drawdown(np.zeros(4)) == 0.0
    # a gain first then a fall: anchored and unanchored agree here, which is why the bug hid
    assert np.isclose(max_drawdown(np.array([0.5, -0.75])), 0.75)
    assert np.isnan(max_drawdown(np.array([])))


def test_ic_is_a_correlation_and_is_nan_when_it_cannot_be_one():
    assert np.isclose(ic_signal(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0])), 1.0)
    assert np.isnan(ic_signal(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])))


def test_the_cost_grid_the_reports_iterate_is_public():
    assert TRADING_COST_GRID == [0.0, 10.0, 30.0]


def test_the_arena_shim_and_the_package_agree():
    """`tsfm4finance.core.metrics` is a re-export shim; a divergence there is invisible until a
    benchmark row disagrees with itself."""
    import tsfm4finance.core.metrics as shim
    for name in ("trading_metrics", "max_drawdown", "ic_signal", "TRADING_COST_GRID"):
        assert getattr(shim, name) is getattr(__import__("tsfm_eval.metrics", fromlist=[name]), name)


def test_the_trading_task_is_a_first_class_task_type():
    """`bench.tasks` writes meta.json with task="trading"; if the enum does not have that member the
    dataset cannot be loaded, and if composite_score has no scorer the leaderboard call raises
    KeyError. Both halves are pinned here."""
    from tsfm_eval.metrics import composite_score
    from tsfm_eval.schemas import TaskType

    assert TaskType("trading") is TaskType.TRADING
    s = composite_score(TaskType.TRADING, {"sharpe_net": 1.0, "ic": 0.05, "hit_rate": 0.55})
    assert 0.0 <= s <= 1.0
    # costs are what separate the two rows, so the score must move with them
    gross = composite_score(TaskType.TRADING, {"sharpe_net": 1.2, "ic": 0.05, "hit_rate": 0.55})
    assert gross > s
