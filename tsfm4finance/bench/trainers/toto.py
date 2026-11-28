"""Toto 2.0 trainer: pinball in the model's own normalised (asinh-scaled) space.

Toto scales with a *causal* scaler, so the training target for the future block is
``asinh((y - loc) / scale)`` computed from context statistics only - the same
transform the decoder inverts at inference.  Anything else would optimise a
different objective than the one the engine is scored on.
"""
from __future__ import annotations

import math
import time

import numpy as np

from tsfm4finance.bench.weights import resolve
from .common import (TrainJob, autocast, count_trainable, free_gpu, log_step,
                     save_ckpt, set_trainable, train_windows)

HEADS = ("output_head", "head")


def train(job: TrainJob) -> dict:
    import torch
    from tsfm4finance.bench.config import load_grid
    from tsfm4finance.bench.engines.toto import Toto2Engine

    spec = load_grid().models()[job.model_key]
    eng = Toto2Engine(checkpoint=resolve(spec.weights), device=job.device, amp="fp32",
                      batch=job.bs, ctx_cap=job.ctx)
    model = eng.model
    model.train()
    trained = set_trainable(model, job.heads_only, HEADS)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=job.effective_lr())
    patch = eng.patch
    pad = int(math.ceil(job.horizon / patch) * patch)

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
        rows, tm_rows, y_rows = [], [], []
        for i in idx:
            c, y = wins[i]
            rows.append(np.concatenate([c, np.zeros(pad)]))
            tm_rows.append(np.r_[np.ones(len(c), dtype=bool), np.zeros(pad, dtype=bool)])
            # the supervision vector has the *same* length as the model input: the
            # context part is ignored by the loss, the future block carries truth
            y_rows.append(np.r_[c, y, np.zeros(pad - job.horizon)])
        t = torch.tensor(np.stack(rows)[:, None, :], dtype=torch.float32, device=job.device)
        tm = torch.tensor(np.stack(tm_rows)[:, None, :], device=job.device)
        yfull = torch.tensor(np.stack(y_rows)[:, None, :], dtype=torch.float32, device=job.device)
        ids = torch.zeros(t.shape[0], 1, dtype=torch.long, device=job.device)
        with autocast(job.device, job.amp):
            # num_return_steps=None keeps the head output aligned with the scaler
            # statistics over the whole (context + future) sequence; we then read
            # the last ``pad`` positions, which are the ones being predicted.
            out = model(target=t, target_mask=tm, cpm_mask=torch.ones_like(tm),
                        series_ids=ids, num_return_steps=None)
        q = out.quantiles.float()                                   # (nq,B,1,P,patch)
        nq = q.shape[0]
        B = t.shape[0]
        q = q.reshape(nq, B, -1).permute(1, 2, 0)                   # (B,P*patch,nq)
        with torch.no_grad():
            _scaled, loc, scale = model.scaler(t, tm & torch.ones_like(tm))
            loc = loc[:, 0, :] if loc.ndim == 3 else loc            # (B,T)
            scale = scale[:, 0, :] if scale.ndim == 3 else scale
        T = q.shape[1]
        loc = loc[:, -T:]
        scale = scale[:, -T:].clamp_min(1e-6)
        tgt = torch.asinh((yfull[:, 0, -T:] - loc) / scale).unsqueeze(-1)   # (B,T,1)
        q = q[:, -T:, :]
        loss = _pinball(q, tgt, eng.levels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        log_step(hist, step, float(loss.item()), t0)

    meta = {"model": job.model_key, "series": job.series, "budget": job.budget,
            "horizon": job.horizon, "ctx": job.ctx, "pad": pad, "amp": job.amp,
            "device": job.device, "trained_parameters": trained,
            "trainable_parameters": count_trainable(model),
            "full_parameters": not job.heads_only, "windows": info, "loss": hist,
            "levels": list(eng.levels), "smoke": job.smoke}
    ck = save_ckpt(job, model, meta)
    free_gpu()
    return {"model": job.model_key, "series": job.series, "budget": job.budget,
            "checkpoint": ck, "steps": n_steps, "final_loss": hist["steps"][-1][1],
            "trainable_parameters": meta["trainable_parameters"], "smoke": job.smoke}


def _pinball(q, y, levels):
    import torch
    lv = torch.tensor([float(x) for x in levels[: q.shape[-1]]],
                      device=q.device, dtype=q.dtype)
    return torch.maximum(lv * (y - q), (lv - 1.0) * (y - q)).mean()
