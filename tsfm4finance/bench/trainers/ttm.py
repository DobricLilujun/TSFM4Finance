"""Granite TTM trainer: HF forward + pinball on the model's own quantile grid.

TTM predicts a *fixed* length (config.prediction_length), so a fine-tune at
horizon h optimises the whole head and the sweep slices the first h steps - that
is what the engine does at inference, so training and scoring agree.
"""
from __future__ import annotations

import time

import numpy as np

from .common import (LEVELS, TrainJob, autocast, combined_loss_t, count_trainable,
                     free_gpu, log_step, save_ckpt, set_trainable, train_windows)

HEADS = ("quantile_head", "prediction_head", "output", "head")


def train(job: TrainJob) -> dict:
    import torch
    from tsfm4finance.bench.config import load_grid
    from tsfm4finance.bench.engines.granite import TTMEngine

    spec = load_grid().models()[job.model_key]
    eng = TTMEngine(checkpoint=spec.weights, device=job.device, amp="fp32",
                    batch=job.bs, ctx_cap=job.ctx)
    model = eng.model
    model.train()
    trained = set_trainable(model, job.heads_only, HEADS)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=job.effective_lr())
    wins, info = train_windows(job.series, job.horizon, job.ctx, job.seed,
                               n_windows=job.budget_kw()["n_windows"])
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
        past = torch.tensor(cs[:, :, None], dtype=torch.float32, device=job.device)
        with autocast(job.device, job.amp):
            out = model(past_values=past)
        q = getattr(out, "quantiles", None)
        mean = out.prediction_outputs.float()
        y = torch.tensor(ys, dtype=torch.float32, device=job.device)
        if q is not None and eng.levels:
            q = q.float()
            if q.ndim == 4:
                q = q[:, :, :, 0]
            q = q.reshape(q.shape[0], -1)[:, :job.horizon, :]
            loss = combined_loss_t(q, y, list(eng.levels))
        else:
            loss = torch.abs(mean.reshape(mean.shape[0], -1)[:, :job.horizon] - y).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        log_step(hist, step, float(loss.item()), t0)

    meta = {"model": job.model_key, "series": job.series, "budget": job.budget,
            "horizon": job.horizon, "ctx": job.ctx, "amp": job.amp, "device": job.device,
            "trained_parameters": trained, "trainable_parameters": count_trainable(model),
            "full_parameters": not job.heads_only, "windows": info, "loss": hist,
            "pred_len": eng.pred_len, "levels": list(eng.levels), "smoke": job.smoke}
    ck = save_ckpt(job, model, meta)
    free_gpu()
    return {"model": job.model_key, "series": job.series, "budget": job.budget,
            "checkpoint": ck, "steps": n_steps, "final_loss": hist["steps"][-1][1],
            "trainable_parameters": meta["trainable_parameters"], "smoke": job.smoke}
