"""TimesFM 2.5-200M trainer: reuses the validated full-parameter loop.

This is the loop that produced the published §4.4 fine-tuned rows (pinball over
the 0.1-0.9 grid + L1 on the median channel, all 231M parameters trainable), so
the multi-model grid stays numerically comparable to the case study.
"""
from __future__ import annotations

import numpy as np

from .common import TrainJob, load_ckpt  # noqa: F401  (contract symmetry)


def train(job: TrainJob) -> dict:
    from tsfm4finance.bench.download.common import load_dataset
    from tsfm4finance.bench.finetune import finetune

    tr, _va, _te, meta = load_dataset(job.series)
    target = str(meta.get("target", "close"))
    close = tr[target].to_numpy(dtype=float)
    k = job.budget_kw()["n_windows"]
    res = finetune(close, out_dir=job.ckpt_path().parent / f"{job.model_key}__{job.series}__{job.budget}_dir",
                   dataset=job.series, k_windows=k, context=job.ctx, target_len=job.horizon,
                   lr=job.effective_lr(), batch_size=job.bs,
                   epochs=(1 if job.smoke else job.effective_epochs()), seed=job.seed,
                   device=job.device, amp=(None if job.amp == "fp32" else job.amp))
    ck = job.ckpt_path()
    src = getattr(res, "ckpt_path", None)
    if src and str(src) != str(ck):
        ck.parent.mkdir(parents=True, exist_ok=True)
        ck.write_bytes(open(src, "rb").read())
    return {"model": job.model_key, "series": job.series, "budget": job.budget,
            "checkpoint": str(ck), "steps": int(getattr(res, "steps", 0)),
            "final_loss": float(getattr(res, "final_loss", float("nan"))),
            "trainable_parameters": int(getattr(res, "trainable_parameters", 0)),
            "full_parameters": True, "smoke": job.smoke}
