"""Chronos-2 trainer: drives the vendor's own ``fit`` (full or LoRA).

Chronos-2's training loop (patching, instance norm, group attention) lives inside
the chronos package; reimplementing it here would silently change what "fine-tuned
Chronos-2" means in the table, so we call ``Chronos2Pipeline.fit`` and export the
resulting weights into the arena checkpoint layout.
"""
from __future__ import annotations

import numpy as np

from .common import TrainJob, free_gpu


def _weights(job, spec) -> str:
    from tsfm4finance.bench.weights import resolve
    return resolve((spec.weights if spec and spec.weights else "amazon/chronos-2"))


def train(job: TrainJob) -> dict:
    import torch
    from chronos import Chronos2Pipeline

    from tsfm4finance.bench.download.common import load_dataset
    from .common import train_windows

    from tsfm4finance.bench.config import load_grid
    spec = load_grid().models().get(job.model_key)
    tr, _va, _te, meta = load_dataset(job.series)
    target = str(meta.get("target", "close"))
    wins, info = train_windows(job.series, job.horizon, job.ctx, job.seed,
                              n_windows=job.budget_kw()["n_windows"], target=target)
    if not wins:
        raise ValueError(f"no train windows for {job.series}")
    # chronos.fit windows the history itself, and only accepts the keys
    # target / past_covariates / future_covariates - so each training sample is
    # handed over as one series of exactly ctx + horizon points.
    inputs = [{"target": torch.tensor(np.r_[w[0], w[1]], dtype=torch.float32)}
              for w in wins]

    from tsfm4finance.bench.weights import resolve
    pipe = Chronos2Pipeline.from_pretrained(resolve(_weights(job, spec)), device_map=job.device,
                                            dtype=torch.float32)
    steps = 2 if job.smoke else int(job.budget_kw()["steps"] or max(
        60, job.effective_epochs() * max(1, len(wins) // max(1, job.bs))))
    out_dir = job.ckpt_path().parent / f"{job.model_key}__{job.series}__{job.budget}_pipe"
    ft = pipe.fit(inputs, prediction_length=job.horizon, finetune_mode="full",
                  context_length=job.ctx, learning_rate=job.effective_lr(),
                  num_steps=steps, batch_size=job.bs, output_dir=out_dir,
                  min_past=min(job.ctx, 32))
    ck = job.ckpt_path()
    torch.save({"model": ft.model.state_dict(),
                "meta": {"model": job.model_key, "series": job.series,
                         "budget": job.budget, "steps": steps, "windows": info,
                         "trainable_parameters": sum(p.numel()
                                                     for p in ft.model.parameters()),
                         "full_parameters": True, "smoke": job.smoke}}, ck)
    free_gpu()
    return {"model": job.model_key, "series": job.series, "budget": job.budget,
            "checkpoint": str(ck), "steps": steps, "n_windows": len(wins),
            "trainable_parameters": sum(p.numel() for p in ft.model.parameters()),
            "full_parameters": True, "smoke": job.smoke}
