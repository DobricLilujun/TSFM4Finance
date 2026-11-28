"""Trainer-contract tests that run without a GPU or vendor weights.

These pin the plumbing bugs the review found: a training window set that silently loses its last
example (empty batch on a short split), a family trainer that would run an incompatible vendor API,
a head-only ablation that quietly trains everything, a few-shot budget that performs zero updates and
saves base weights as a fine-tune, and a CLI that exits 0 after failures so the runner and Slurm
believe the checkpoints exist.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tsfm4finance.bench.trainers.common import TrainJob, train_windows


def _fake_splits(monkeypatch, n_train: int, n_feat: int = 1):
    from tsfm4finance.bench.download import common as dl

    idx = pd.date_range("2020-01-01", periods=n_train, freq="D")
    tr = pd.DataFrame({"timestamp": idx, "close": np.arange(n_train, dtype=float)})
    empty = tr.iloc[:0]
    meta = {"target": "close", "n_obs": n_train}

    def load_dataset(series):
        return tr, empty, empty, meta
    monkeypatch.setattr(dl, "load_dataset", load_dataset)
    return tr


def test_the_last_legal_training_window_is_not_off_by_one(tmp_path, monkeypatch):
    """A train split of exactly ctx + horizon observations holds exactly one supervised example.
    The exclusive upper bound returned no windows at all, and the family trainers then died on an
    empty batch - a data-shape failure reported as a model failure."""
    ctx, h = 64, 21
    _fake_splits(monkeypatch, ctx + h)
    wins, info = train_windows("any-series", horizon=h, ctx=ctx, n_windows=None)
    assert len(wins) == 1, f"expected exactly one window, got {len(wins)}"
    c, y = wins[0]
    assert len(c) == ctx and len(y) == h
    # one more observation must give exactly one more window, not zero
    _fake_splits(monkeypatch, ctx + h + 1)
    wins2, _ = train_windows("any-series", horizon=h, ctx=ctx, n_windows=None)
    assert len(wins2) == 2


def test_chronos2_trainer_refuses_variants_it_cannot_train(monkeypatch):
    """The registry routes the whole "chronos" family to this loop, but only Chronos-2 has the
    vendor `fit` API it calls; Bolt and T5 would have run an incompatible trainer."""
    from tsfm4finance.bench.trainers import chronos2

    job = TrainJob(model_key="chronos-bolt-small", series="s", budget="fs16")
    out = chronos2.train(job)
    assert out["status"] == "trainer-pending", out
    assert "Bolt" in out["reason"] or "chronos" in out["reason"]


def test_heads_only_is_rejected_by_trainers_that_cannot_honour_it():
    """A frozen-backbone ablation that trains the whole model is not an ablation; the cell would land
    in the same column as a real head-only run."""
    from tsfm4finance.bench.trainers.__main__ import run_one

    job = TrainJob(model_key="timesfm-2.5-200m", series="s", budget="full", heads_only=True)
    out = run_one(job, "timesfm25")
    assert out["status"] == "rejected", out
    assert "heads_only" in out["reason"]


def test_a_step_budget_is_not_zero_epochs(monkeypatch, tmp_path):
    """fs16/fs64 specify optimizer steps, so their budget has epochs == 0. The trainer passed that
    straight through: both few-shot runs trained zero steps and saved base weights as "fine-tuned"."""
    from tsfm4finance.bench.trainers import timesfm25
    from tsfm4finance.bench.download import common as dl

    idx = pd.date_range("2020-01-01", periods=800, freq="D")
    tr = pd.DataFrame({"timestamp": idx, "close": np.arange(800, dtype=float)})
    monkeypatch.setattr(dl, "load_dataset",
                        lambda s: (tr, tr.iloc[:0], tr.iloc[:0], {"target": "close"}))

    seen = {}

    class _Res:
        checkpoint = str(tmp_path / "dir")
        ckpt_file = str(tmp_path / "state_dict.pt")
        steps = 120
        final_loss = 0.5
        trainable_parameters = 231_000_000

    (tmp_path / "state_dict.pt").write_bytes(b"x")

    def fake_finetune(train_close, out_dir, **kw):
        seen.update(kw)
        return _Res()

    monkeypatch.setattr(timesfm25, "finetune", fake_finetune, raising=False)
    monkeypatch.setattr("tsfm4finance.bench.finetune.finetune", fake_finetune)

    job = TrainJob(model_key="chronos-2" and "timesfm-2.5-200m", series="s", budget="fs64",
                   ctx=256, horizon=21, bs=8, out_dir=str(tmp_path))
    assert job.effective_epochs() == 0, "the fs64 budget is step-based by construction"
    res = timesfm25.train(job)
    assert seen["epochs"] >= 1, f"zero-epoch fine-tune reached the trainer: {seen}"
    assert res["checkpoint"].endswith(".pt")
    assert (tmp_path / "timesfm-2.5-200m__s__fs64.pt").exists(), "adapted cells are discovered by file"
