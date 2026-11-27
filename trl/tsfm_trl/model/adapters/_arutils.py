"""Shared plumbing for the autoregressive "continuous token" family (Time-MoE,
Sundial, and anything else that takes float values as tokens and regresses the next
value).

Their heads predict *values*, not quantiles, so the training signal is a Huber loss
in the model's own per-window normalised space — the same normalisation the
inference path uses. If a checkpoint turns out to use a discrete vocabulary, this
raises :class:`CapabilityError` with the reason instead of training the wrong thing.
"""
from __future__ import annotations

import numpy as np

from ...errors import CapabilityError


def normalise(cs, ys):
    x = np.asarray(cs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)
    mu = x.mean(axis=1, keepdims=True)
    sd = np.where(x.std(axis=1, keepdims=True) > 0, x.std(axis=1, keepdims=True), 1.0)
    return (x - mu) / sd, (y - mu) / sd


def ar_value_step(model, contexts, truth, horizon, *, device: str, key: str,
                  attn: bool = True):
    """One Huber step on the normalised future values.

    ``model`` is the HF causal model; ``input_ids`` are normalised *values* (that is
    literally how these checkpoints are wired). Returns a scalar loss tensor.
    """
    import torch
    from ...training.losses import huber_point
    xn, yn = normalise(contexts, truth)
    full = np.concatenate([xn, yn], axis=1).astype(np.float32)
    t = torch.tensor(full, device=device, dtype=torch.float32)
    mask = torch.ones_like(t, dtype=torch.long) if attn else None
    try:
        out = model(input_ids=t, attention_mask=mask) if attn else model(input_ids=t)
    except TypeError:
        out = model(t)
    logits = getattr(out, "logits", None)
    if logits is None:
        raise CapabilityError(key, "finetune", "model forward returned no logits")
    logits = logits.float()
    H = yn.shape[1]
    if logits.shape[-1] == 1:
        pred = logits[:, -(H + 1):-1, 0]                     # next-value head
        if pred.shape[1] != H:
            pred = logits[:, -H:, 0]
        tgt = torch.tensor(yn, device=device, dtype=torch.float32)
        scale = max(1e-6, float(np.nanmean(np.abs(contexts))) and 1.0)
        return huber_point(pred, tgt, beta=1.0)
    raise CapabilityError(
        key, "finetune",
        f"head emits a discrete vocabulary ({logits.shape[-1]} classes); fine-tuning needs "
        f"the vendor tokenizer ids — wire them in the adapter rather than regressing values")
