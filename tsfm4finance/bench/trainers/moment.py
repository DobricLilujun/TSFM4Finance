"""MOMENT-1 trainer: forecasting head + backbone, L1 on the point forecast.

MOMENT's forecasting head emits a point forecast (no quantile grid), so the
training loss is L1 on the horizon block; the benchmark still scores it with the
same point metrics (MASE/NDp/DM) and simply has no probabilistic columns.
"""
from __future__ import annotations

import time

import numpy as np

from tsfm4finance.bench.weights import resolve
from .common import (TrainJob, autocast, count_trainable, free_gpu, log_step,
                     save_ckpt, set_trainable, train_windows)

HEADS = ("head", "projection", "forecast")


def train(job: TrainJob) -> dict:
    import torch
    from momentfm import MOMENTPipeline
    from tsfm4finance.bench.config import load_grid

    spec = load_grid().models()[job.model_key]
    p = MOMENTPipeline.from_pretrained(resolve(spec.weights),
                                       model_kwargs={"task_name": "forecasting",
                                                     "n_channels": 1,
                                                     "forecast_horizon": job.horizon})
    p.init()
    p = p.to(job.device)
    p.train()
    trained = set_trainable(p, job.heads_only, HEADS)
    opt = torch.optim.AdamW([q for q in p.parameters() if q.requires_grad],
                            lr=job.effective_lr())

    # MOMENT's forecast head is a dense map from (context/patch_len * d_model) to
    # the horizon, so the context length is part of the checkpoint: crop/pad to it
    # instead of letting the matmul fail.
    want = int(getattr(getattr(p, "config", None), "context_seq_len", 0)
               or getattr(getattr(p, "module", None), "config", None) and
               getattr(p.module.config, "context_seq_len", 0) or 512)
    ctx_use = min(max(job.ctx, 64), want) if want else job.ctx
    wins, info = train_windows(job.series, job.horizon, ctx_use, job.seed,
                               n_windows=job.budget_kw()["n_windows"])
    if want and ctx_use < want:
        wins = [(np.r_[np.full(want - ctx_use, c[0]), c], y) for c, y in wins]
        info["context_padded_to"] = want
    rng = np.random.RandomState(job.seed)
    n_steps = 2 if job.smoke else int(job.budget_kw()["steps"] or
                                      max(60, job.effective_epochs() *
                                          max(1, len(wins) // max(1, job.bs))))
    hist: dict = {}
    t0 = time.time()
    for step in range(1, n_steps + 1):
        idx = rng.choice(len(wins), size=min(job.bs, len(wins)), replace=False)
        cs = np.stack([wins[i][0] for i in idx])
        ys = np.stack([wins[i][1] for i in idx])
        x = torch.tensor(cs[:, None, :], dtype=torch.float32, device=job.device)
        y = torch.tensor(ys[:, None, :], dtype=torch.float32, device=job.device)
        with autocast(job.device, job.amp):
            out = p(x_enc=x)
        pred = getattr(out, "forecast", None)
        if pred is None:
            pred = getattr(out, "prediction", None)
        pred = pred.float()
        if pred.shape[-1] < job.horizon:
            pred = torch.nn.functional.pad(pred, (0, job.horizon - pred.shape[-1]))
        loss = torch.abs(pred[..., :job.horizon] - y).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(p.parameters(), 1.0)
        opt.step()
        log_step(hist, step, float(loss.item()), t0)

    meta = {"model": job.model_key, "series": job.series, "budget": job.budget,
            "horizon": job.horizon, "ctx": job.ctx, "amp": job.amp, "device": job.device,
            "trained_parameters": trained, "trainable_parameters": count_trainable(p),
            "full_parameters": not job.heads_only, "windows": info, "loss": hist,
            "smoke": job.smoke}
    ck = save_ckpt(job, p, meta)
    free_gpu()
    return {"model": job.model_key, "series": job.series, "budget": job.budget,
            "checkpoint": ck, "steps": n_steps, "final_loss": hist["steps"][-1][1],
            "trainable_parameters": meta["trainable_parameters"], "smoke": job.smoke}
