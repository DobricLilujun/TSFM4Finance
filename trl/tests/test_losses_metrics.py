"""Losses and metrics are the parts that are easy to get subtly wrong and expensive
to discover later (a pinball loss with the wrong level convention makes every
interval metric wrong). These are exact-value checks."""
from __future__ import annotations

import numpy as np
import pytest

from tsfm_trl.eval.metrics import (classification_report, crps_from_quantiles, mae, mase,
                                   pinball, quantile_coverage, relative_skill, rmse, smape, wql)
from tsfm_trl.model.base import LEVELS, quantiles_from_samples, to_levels
from tsfm_trl.training.losses import quantile_loss, reward_from_quantiles


def test_pinball_matches_the_textbook_value():
    # y=10, q_hat=9, tau=0.9 -> 0.9*(10-9) = 0.9 ; over-prediction costs (1-tau)
    assert np.isclose(pinball(np.array([[10.0]]), np.array([[[9.0]]]), [0.9]), 0.9)
    assert np.isclose(pinball(np.array([[10.0]]), np.array([[[11.0]]]), [0.9]), 0.1)


def test_pinball_is_the_mean_over_levels():
    import torch
    q = torch.tensor([[[8.0, 9.0, 10.0]]])
    y = torch.tensor([[9.5]])
    lv = [0.1, 0.5, 0.9]
    manual = float(np.mean([pinball(y.numpy(), q[..., j, None].numpy(), [lv[j]])
                            for j in range(3)]))
    # w_point=0, w_cross=0 isolates the pure pinball component
    assert np.isclose(quantile_loss(q, y, lv, 0.0, 0.0).item(), manual, rtol=1e-6)


def test_quantile_loss_prefers_a_tighter_correct_interval():
    import torch
    y = torch.tensor([[0.0, 1.0, -0.5]])
    wide = np.stack([np.linspace(-6, 6, len(LEVELS))]).reshape(1, 1, -1).repeat(3, 1)
    tight = np.stack([np.linspace(-0.8, 1.2, len(LEVELS))]).reshape(1, 1, -1).repeat(3, 1)
    assert quantile_loss(torch.tensor(tight), y, LEVELS) < quantile_loss(torch.tensor(wide), y, LEVELS)


def test_point_metrics_known_values():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0.0 and rmse(y, y) == 0.0
    assert np.isclose(rmse(np.zeros(4), np.array([3.0, -3.0, 3.0, -3.0])), 3.0)
    assert np.isclose(smape(np.array([10.0, 10.0]), np.array([11.0, 9.0])), 10.025, atol=0.01)
    train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.isclose(mase(train[-3:], train[-3:], train), 0.0)
    assert np.isclose(mase(train[-3:], train[:-1][-3:], train), 1.0, atol=1e-9)


def test_relative_skill_direction():
    y = np.arange(10.0)
    assert relative_skill(y, y, y) == pytest.approx(1.0)                    # perfect
    assert relative_skill(y, y[:-1], y) == pytest.approx(0.0)               # = naive
    assert relative_skill(y, y + 5.0, y) < 0.0                              # worse


def test_quantile_grid_helpers():
    rng = np.random.RandomState(0)
    q = np.sort(rng.normal(size=(2, 3, 5)), axis=-1)
    src = [0.2, 0.4, 0.5, 0.6, 0.8]
    out = to_levels(q, src)
    assert out.shape == (2, 3, len(LEVELS))
    assert np.all(np.diff(out, axis=-1) >= -1e-9), "levels must stay monotone"
    med = out[:, :, LEVELS.index(0.5)]
    assert np.allclose(med, q[:, :, 2], atol=1e-6)      # 0.5 exists in src -> exact


def test_samples_to_quantiles():
    # samples are (series, path, horizon); the output must be (series, horizon, level)
    s = np.arange(100.0).reshape(1, 100, 1).repeat(2, 0)
    q = quantiles_from_samples(s)
    assert q.shape == (2, 1, len(LEVELS)), q.shape
    assert q[0, 0, LEVELS.index(0.5)] == pytest.approx(49.5, rel=1e-2)


def test_coverage_and_wql_are_sane():
    rng = np.random.RandomState(1)
    y = rng.normal(size=(50, 5))
    q = np.stack([np.full_like(y, np.quantile(y, lv)) for lv in LEVELS], axis=-1)
    q = np.sort(q, axis=-1)
    cov = quantile_coverage(y, q)
    # with the 0.1..0.9 grid the widest band the model *emits* is 80 %, not 90 %
    assert cov["coverage_80"] >= cov["coverage_60"]
    assert cov["nominal_80"] == pytest.approx(0.8)
    assert "coverage_95" not in cov
    assert np.isfinite(wql(y, q, LEVELS))
    crossed = q[:, :, ::-1].copy()                      # deliberately non-monotone
    assert wql(y, crossed, LEVELS) >= wql(y, q, LEVELS)


def test_crps_from_quantiles_is_nonnegative_and_finite():
    rng = np.random.RandomState(2)
    y = rng.normal(size=(20, 4))
    q = np.stack([y + lv * 0.5 for lv in LEVELS], axis=-1)
    c = crps_from_quantiles(y, q, LEVELS)
    assert np.all(c >= 0) and np.all(np.isfinite(c))


def test_reward_rewards_a_correct_forecast():
    import torch
    y = torch.ones(2, 4)
    good = np.stack([np.full((2, 4), v) for v in np.linspace(0.9, 1.1, len(LEVELS))], axis=-1)
    bad = np.stack([np.full((2, 4), v) for v in np.linspace(0.0, 2.0, len(LEVELS))], axis=-1)
    assert reward_from_quantiles(torch.tensor(good), y).mean() > \
        reward_from_quantiles(torch.tensor(bad), y).mean()


def test_classification_report_shapes():
    y = np.array([0, 1, 1, 0, 1])
    score = np.array([0.1, 0.9, 0.6, 0.4, 0.55])
    rep = classification_report(y, score)
    for k in ("pr_auc", "roc_auc", "macro_f1", "n", "n_classes"):
        assert k in rep, rep
    assert rep["roc_auc"] == pytest.approx(1.0)
    assert rep["n"] == 5 and rep["n_classes"] == 2


def test_classification_report_multiclass_scores():
    """PR-AUC/ROC-AUC exist for >2 classes too (one-vs-rest weighted), and only when the
    caller passes scores — hard labels must report nan, not a fabricated number."""
    y = np.array([0, 0, 1, 1, 2, 2, 2])
    rng = np.random.RandomState(0)
    s = rng.dirichlet([0.6] * 3, size=7) * 0.4
    s[np.arange(7), y] += 0.6                          # a classifier that is right, mostly
    rep = classification_report(y, s, n_classes=3)
    assert rep["accuracy"] == 1.0, rep
    assert np.isfinite(rep["pr_auc"]) and np.isfinite(rep["roc_auc"]), rep
    assert 0.0 <= rep["pr_auc"] <= 1.0 and 0.0 < rep["roc_auc"] <= 1.0
    assert np.isfinite(rep["f1_beta2"]), rep           # macro one-vs-rest F2
    hard = classification_report(y, y, n_classes=3)    # perfect labels, no scores
    assert hard["accuracy"] == 1.0 and np.isnan(hard["pr_auc"]), hard
    assert np.isfinite(hard["macro_f1"])


def test_classification_report_degenerate_split_does_not_raise():
    """A matrix run can hand back a fold with one class; that is a data fact, not a bug."""
    y = np.array([1, 1, 1])
    rep = classification_report(y, np.array([0.6, 0.7, 0.5]), n_classes=2)
    assert rep["n_classes"] == 2 and np.isnan(rep["pr_auc"])
    assert np.isfinite(rep["macro_f1"]) and rep["base_rate"] == 1.0


def test_to_numpy_moves_vendor_gpu_tensors():
    """Vendor forecasters hand back CUDA tensors; the matrix must not file that as a
    failure of the model."""
    import numpy as np
    import torch
    from tsfm_trl.model.base import to_numpy
    assert np.allclose(to_numpy(torch.zeros(3)), np.zeros(3))
    if torch.cuda.is_available():
        try:
            t = torch.ones(2, device="cuda")
        except RuntimeError:                        # a busy GPU: not a reason to fail
            t = torch.ones(2)
        assert to_numpy(t).shape == (2,)
    assert to_numpy(np.arange(4)).dtype == np.float64
    assert to_numpy(None) is None


def test_dedupe_keeps_the_working_row_and_notes_the_other_machine():
    from tsfm_trl.doctor import Row, dedupe
    a = Row(model="x", env="core", load="ok", predict="ok", machine="laptop")
    b = Row(model="x", env="core", load="fail", predict="skip", machine="gpu7",
            reasons={"load": "triton kernel"})
    out = dedupe([a, b])
    assert len(out) == 1 and out[0].load == "ok"
    assert "gpu7" in out[0].reasons["other machine"], out[0].reasons
    assert dedupe([a])[0] is a
