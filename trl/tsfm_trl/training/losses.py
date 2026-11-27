"""Losses. One shared contract: everything is computed in **real units** unless the
adapter says otherwise, because financial series differ by orders of magnitude and
a loss that silently depends on scale makes two models incomparable.

* ``pinball_loss`` / ``quantile_loss`` — the probabilistic forecast head family
  (Chronos-Bolt/-2, Moirai, TimesFM, Toto, TiRex, Timestral, t0, Hirondel, …).
* ``token_cross_entropy`` — the autoregressive-token family (Chronos-T5, Sundial,
  Time-MoE, Kronos): supervised next-token loss on the realised future.
* ``flow_matching_loss`` — Aurora's head (its own noise schedule; see the adapter).
* ``quantile_crossing_penalty`` — regulariser: released heads are occasionally
  non-monotone at the tails on short financial windows.
"""
from __future__ import annotations

import numpy as np

LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def pinball_loss(q, y, levels=LEVELS):
    """Pinball on ``q`` (B,H,nq) vs ``y`` (B,H)."""
    import torch
    lv = torch.tensor(list(levels), device=q.device, dtype=q.dtype)
    y = y.unsqueeze(-1)
    return torch.maximum(lv * (y - q), (lv - 1.0) * (y - q)).mean()


def quantile_crossing_penalty(q):
    import torch
    d = q[..., 1:] - q[..., :-1]
    return torch.relu(-d).mean()


def quantile_loss(q, y, levels=LEVELS, w_point: float = 1.0, w_cross: float = 0.1):
    """Pinball + L1 on the median channel + a monotonicity penalty.

    The extra L1 term is not decoration: pure pinball lets the median drift between
    grid levels on short windows, which shows up later as a directional bias.
    """
    import torch
    if q is None:
        raise ValueError("model returned no quantile head")
    lv = list(levels)
    med = int(np.argmin(np.abs(np.asarray(lv, float) - 0.5)))
    loss = pinball_loss(q, y, lv) + w_point * torch.abs(q[..., med] - y).mean()
    if w_cross:
        loss = loss + w_cross * quantile_crossing_penalty(q)
    return loss


def token_cross_entropy(logits, target_ids, ignore_index: int = -100):
    """Next-token CE for AR-token forecast models.

    ``logits`` (B,T,V) predicts ``target_ids`` (B,T) shifted by one position, so the
    caller passes the *full* context+future token sequence.
    """
    import torch
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = target_ids[:, 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1), ignore_index=ignore_index)


def huber_point(pred, y, beta: float = 1.0):
    import torch
    return torch.nn.functional.huber_loss(pred, y, delta=beta)


def flow_matching_loss(pred_v, x_target, x_context, noise, t, levels=None):
    """Aurora-style conditional flow matching: predict the velocity field at time t
    between a noisy source and the target window."""
    import torch
    return torch.nn.functional.mse_loss(pred_v, x_target - noise)


def reward_from_quantiles(q, y, levels=LEVELS, w_pinball: float = 1.0,
                          w_dir: float = 0.5, w_cross: float = 0.2, anchor=None):
    """Differentiable *reward* (higher = better) used by the reward/RL trainer when
    the model exposes a quantile head:

        reward = -w_pinball * pinball  +  w_dir * direction  -  w_cross * crossing

    ``anchor`` (B,1) is the last context value: without it the direction term is
    meaningless, because a level forecast says nothing about up/down. Kept here so
    the reward definition is shared and unit-testable."""
    import torch
    lv = list(levels)
    med = int(np.argmin(np.abs(np.asarray(lv, float) - 0.5)))
    r = -w_pinball * pinball_loss(q, y, lv) - w_cross * quantile_crossing_penalty(q)
    if anchor is not None:
        a = torch.as_tensor(anchor).reshape(q.shape[0], *([1] * (q[..., med].dim() - 1)))
        r = r + w_dir * (torch.sign(q[..., med] - a) * torch.sign(y - a)).mean()
    return r


def directional_reward(pred, y):
    """+1 where the sign of the change (relative to the last context value) is
    right, −1 where it is wrong. Not differentiable — a policy-gradient term."""
    import torch
    d_pred = torch.sign(pred)
    d_true = torch.sign(y)
    return (d_pred * d_true).mean()
