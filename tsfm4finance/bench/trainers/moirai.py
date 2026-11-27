"""Moirai 1.1 / 2.0 / Moirai-MoE trainer: gradient steps through the same
predictor forward the engine uses at inference, with pinball on the model's own
quantile grid (real space, i.e. after its internal scaler)."""
from __future__ import annotations

import numpy as np

from .common import TrainJob, autocast, count_trainable, free_gpu, log_step, save_ckpt, set_trainable, train_windows

HEADS = ("p21", "quantile_head", "head", "output_proj", "distribution_output")


def train(job: TrainJob) -> dict:
    import torch
    from tsfm4finance.bench.engines.moirai import MoiraiEngine
    from .common import LEVELS, combined_loss_t

    eng = MoiraiEngine(checkpoint=_weights(job), device=job.device, amp="fp32",
                       batch=job.bs, ctx_cap=job.ctx)
    f = eng._F(module=eng._mod, prediction_length=job.horizon, target_dim=1,
               feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
               context_length=job.ctx)
    f = f.to(job.device)
    f.train()
    trained = set_trainable(f, job.heads_only, HEADS)
    opt = torch.optim.AdamW([p for p in f.parameters() if p.requires_grad],
                            lr=job.effective_lr())

    wins, info = train_windows(job.series, job.horizon, job.ctx, job.seed,
                               n_windows=job.budget_kw()["n_windows"])
    rng = np.random.RandomState(job.seed)
    n_steps = 2 if job.smoke else int(job.budget_kw()["steps"] or
                                      max(60, job.effective_epochs() *
                                          max(1, len(wins) // max(1, job.bs))))
    hist: dict = {}
    t0 = __import__("time").time()
    for step in range(1, n_steps + 1):
        cs, ys = _batch(wins, job.bs, rng)
        pt = torch.tensor(cs[:, :, None], dtype=torch.float32, device=job.device)
        obs = torch.ones_like(pt, dtype=torch.bool)
        pad = torch.zeros(pt.shape[:2], dtype=torch.bool, device=job.device)
        with autocast(job.device, job.amp):
            q = f(pt, obs, pad)
        q = q.float()
        if q.ndim == 4:
            q = q[:, :, :, 0]
        q = q.transpose(1, 2)[:, :job.horizon, :]            # (B,H,nq)
        lv = eng._levels[: q.shape[-1]]
        loss = combined_loss_t(q, torch.tensor(ys, device=job.device), lv)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(f.parameters(), 1.0)
        opt.step()
        log_step(hist, step, float(loss.item()), t0)

    meta = {"model": job.model_key, "series": job.series, "budget": job.budget,
            "horizon": job.horizon, "ctx": job.ctx, "amp": job.amp, "device": job.device,
            "trained_parameters": trained, "trainable_parameters": count_trainable(f),
            "full_parameters": not job.heads_only, "windows": info, "loss": hist,
            "levels": list(eng._levels), "smoke": job.smoke}
    ck = save_ckpt(job, f, meta)
    free_gpu()
    return {"model": job.model_key, "series": job.series, "budget": job.budget,
            "checkpoint": ck, "steps": n_steps, "final_loss": hist["steps"][-1][1],
            "trainable_parameters": meta["trainable_parameters"], "smoke": job.smoke}


def _weights(job: TrainJob) -> str:
    from tsfm4finance.bench.config import load_grid
    spec = load_grid().models().get(job.model_key)
    if spec is None or not spec.weights:
        raise KeyError(f"no weights configured for {job.model_key}")
    return spec.weights


def _batch(wins, bs, rng):
    idx = rng.choice(len(wins), size=min(bs, len(wins)), replace=False)
    return (np.stack([wins[i][0] for i in idx]), np.stack([wins[i][1] for i in idx]))
