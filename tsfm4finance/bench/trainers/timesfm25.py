"""TimesFM 2.5-200M trainer: reuses the validated full-parameter loop.

This is the loop that produced the published §4.4 fine-tuned rows (pinball over
the 0.1-0.9 grid + L1 on the median channel, all 231M parameters trainable), so
the multi-model grid stays numerically comparable to the case study.
"""
from __future__ import annotations

import numpy as np

from .common import TrainJob, load_ckpt  # noqa: F401  (contract symmetry)


# this trainer updates the whole model: it cannot honour job.heads_only, so the CLI rejects
# that combination instead of labelling a full-model run as an ablation
HEADS_ONLY = False

def train(job: TrainJob) -> dict:
    from tsfm4finance.bench.download.common import load_dataset
    from tsfm4finance.bench.finetune import finetune

    tr, _va, _te, meta = load_dataset(job.series)
    target = str(meta.get("target", "close"))
    close = tr[target].to_numpy(dtype=float)
    k = job.budget_kw()["n_windows"]
    # fs16 / fs64 specify optimizer *steps*, so their budget has epochs == 0. Passing that straight
    # through meant both few-shot runs trained for zero steps and saved base weights as if they were
    # fine-tuned results. Convert the step budget into epochs over the window set instead.
    ep = 1 if job.smoke else job.effective_epochs()
    steps_budget = int(job.budget_kw().get("steps") or 0)
    if ep <= 0 and steps_budget > 0:
        per_epoch = max(1, -(-k // job.bs))            # ceil(k / batch): batches per epoch
        ep = max(1, -(-steps_budget // per_epoch))
    res = finetune(close, out_dir=job.ckpt_path().parent / f"{job.model_key}__{job.series}__{job.budget}_dir",
                   dataset=job.series, k_windows=k, context=job.ctx, target_len=job.horizon,
                   lr=job.effective_lr(), batch_size=job.bs,
                   epochs=ep, seed=job.seed,
                   device=job.device, amp=(None if job.amp == "fp32" else job.amp))
    if int(getattr(res, "steps", 0)) == 0:
        raise RuntimeError("timesfm25 fine-tune performed zero updates; refusing to export "
                           "base weights as a fine-tuned checkpoint")
    ck = job.ckpt_path()
    # the worker and the planner discover adapted cells through this file, so it must be the
    # state-dict form - copying the save_pretrained directory's bytes here would not load
    src = getattr(res, "ckpt_file", None)
    if src and str(src) != str(ck):
        ck.parent.mkdir(parents=True, exist_ok=True)
        ck.write_bytes(open(src, "rb").read())
    return {"model": job.model_key, "series": job.series, "budget": job.budget,
            "checkpoint": str(ck), "steps": int(getattr(res, "steps", 0)),
            "final_loss": float(getattr(res, "final_loss", float("nan"))),
            "trainable_parameters": int(getattr(res, "trainable_parameters", 0)),
            "full_parameters": True, "smoke": job.smoke}
