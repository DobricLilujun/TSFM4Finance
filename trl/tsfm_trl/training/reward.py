"""``RewardTrainer`` — reward-based post-training (the "RL" half of TRL).

Three modes, chosen by what the model can do, all with the same reward definition
(see :func:`tsfm_trl.training.losses.reward_from_quantiles` and :class:`RewardSpec`):

``reward_descent``
    Gradient ascent on a *differentiable* reward through the model's own quantile
    head: −pinball + directional term − quantile-crossing penalty. The most direct
    analogue of reward fine-tuning for probabilistic forecasters.

``rft``  (rejection-sampling fine-tuning / best-of-N)
    Draw K candidate futures from the model's own predictive distribution (inverse
    CDF on its quantile grid, or its real sampled paths when it has them), score
    every candidate with the reward, keep the group-best (advantage > group mean),
    and take a supervised step toward it. This is policy-gradient-with-group-baseline
    in the limit, needs no token log-probs, and works for every quantile model.

``grpo`` (true group-relative policy gradient)
    For autoregressive-token adapters that expose ``tokenify`` + ``path_logprob``
    (Chronos-T5 today): sample K continuations, group-normalise the rewards into
    advantages, and do the usual ``-A * logp`` update with a KL anchor to the
    reference weights.

The reward is *not* fixed: pass ``reward=`` any callable
``f(candidates (B,K,H), truth (B,H), anchor (B,1)) -> (B,K)`` — e.g. a trading
reward (signal × realised return minus turnover) — and everything else adapts.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np

from ..errors import CapabilityError
from ..model.base import LEVELS
from .config import StepLog, TrainResult, TrainingConfig
from .peft import count_trainable, move_deep


@dataclass
class RewardSpec:
    """Default reward: accuracy + direction, minus the pathologies that make a
    probabilistic forecast useless in a portfolio (crossing quantiles, no tails)."""

    w_pinball: float = 1.0
    w_dir: float = 0.5
    w_cross: float = 0.2
    clip: float = 50.0                   # keep rewards finite on outlier windows

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def reward_candidates(q_pred, cand, truth, anchor, spec: RewardSpec):
    """Score ``cand`` (B,K,H) against ``truth`` (B,H), given the model's quantiles
    for scale. Higher = better."""
    import torch
    d = (truth.unsqueeze(1) - cand)
    scale = (q_pred[..., -1] - q_pred[..., 0]).unsqueeze(1).clamp_min(1e-6) if q_pred.shape[-1] > 1 \
        else cand.abs().mean(-1, keepdim=True).clamp_min(1e-6)
    point = -d.abs() / scale                                   # scale-free accuracy
    dirn = torch.sign(cand - anchor.unsqueeze(1)) * torch.sign(truth.unsqueeze(1) - anchor.unsqueeze(1))
    r = -spec.w_pinball * point + spec.w_dir * dirn
    return r.clamp(-spec.clip, spec.clip).mean(-1)             # (B,K)


class RewardTrainer:
    def __init__(self, model, cfg: TrainingConfig | None = None, *, mode: str = "auto",
                 reward: "callable | None" = None, n_candidates: int = 8,
                 kl: float = 1e-3, spec: RewardSpec | None = None, **cfg_kw):
        self.model = model
        self.cfg = cfg or TrainingConfig(**cfg_kw)
        self.mode_pref = mode
        self.reward = reward
        self.n_candidates = int(n_candidates)
        self.kl = float(kl)
        self.spec = spec or RewardSpec()

    # ------------------------------------------------------------------ data
    def _windows(self, data):
        from .forecast import ForecastTrainer
        return ForecastTrainer(self.model, self.cfg)._windows(data)

    def fit(self, data) -> TrainResult:
        """Reward post-training with the same memory ladder as the forecast trainer.

        Profiles are tried in order of invasiveness; every fallback is recorded in
        the checkpoint, so a laptop result says "reward-trained with heads_only at
        bs=1 bf16" instead of looking identical to a full-parameter run on a 48 GB card.
        """
        from .forecast import _free, _is_env, _is_oom
        windows, info = self._windows(data)
        key = getattr(self.model, "key", "model")
        # reward post-training needs a differentiable model. An adapter that is
        # inference-only (by licence or by vendor release) should read "not offered",
        # not a training failure after four empty PEFT profiles.
        if "finetune" not in (getattr(self.model, "caps", None) or ()):
            sup = getattr(self.model, "supports", {}) or {}
            why = sup.get("finetune") if isinstance(sup.get("finetune"), str) else \
                "adapter exposes no trainable module"
            return TrainResult(model=key, status="skipped",
                               reason=f"{key}: no 'finetune' support ({why})")
        base = (self.cfg.bs, self.cfg.amp, self.cfg.peft)
        profiles = [base]
        if base != (1, "bf16", base[2]):
            profiles.append((1, "bf16", base[2]))
        for peft in ("heads_only", "lora"):
            if peft != base[2]:
                profiles.append((1, "bf16", peft))
        profiles.append((1, "fp32", "heads_only"))
        last = None
        for i, (bs, amp, peft) in enumerate(profiles):
            try:
                return self._attempt(windows, info, bs=bs, amp=amp, peft=peft,
                                     fallbacks=[f"bs={bs},amp={amp},peft={peft}"]
                                               if i else [])
            except CapabilityError as e:
                if "no '" in str(e) and "' support" in str(e):
                    # the model itself declines (inference-only release, no trainable
                    # module); no PEFT profile changes that
                    return TrainResult(model=key, status="skipped", reason=str(e),
                                       n_windows=len(windows))
                # "heads_only matched nothing" is a *profile* problem, not a verdict on
                # the model: the next profile (lora) may well work
                if i + 1 < len(profiles):
                    last = e
                    continue
                return TrainResult(model=key, status="skipped", reason=str(e),
                                   n_windows=len(windows))
            except Exception as e:                                    # noqa: BLE001
                last = e
                if not (_is_oom(e) or _is_env(e)):
                    return TrainResult(model=key, status="failed",
                                       reason=f"{type(e).__name__}: {e}"[:400],
                                       n_windows=len(windows))
                _free()
        return TrainResult(model=key, status="failed",
                           reason=f"reward training failed even on the last profile ({last})"[:400],
                           n_windows=len(windows))

    def _attempt(self, windows, info, *, bs, amp, peft, fallbacks) -> TrainResult:
        import torch
        cfg = self.cfg
        model = self.model
        key = getattr(model, "key", "model")
        t0 = time.time()
        mode = self.mode_pref
        if mode == "auto":
            if getattr(model, "loss_kind", None) == "token" and hasattr(model, "path_logprob"):
                mode = "grpo"
            elif hasattr(model, "forward_quantiles") and getattr(model, "levels", None):
                mode = "reward_descent"
            elif _has_point_head(model):
                mode = "reward_descent_point"
            elif hasattr(model, "training_step"):
                mode = "rft"
            else:
                return TrainResult(model=key, status="skipped",
                                   reason="no reward path: model exposes neither a forecast step "
                                          "(rft), a differentiable quantile head "
                                          "(reward_descent) nor token log-probs (grpo)")
        model.prepare_training(cfg.ctx, cfg.horizon)
        module = model.trainable()
        if module is None:
            raise CapabilityError(key, "finetune", "adapter exposes no trainable module")
        dev = "cuda" if (cfg.device != "cpu" and torch.cuda.is_available()) else "cpu"
        if dev == "cpu":
            amp = "fp32"
        cfg = replace(cfg, bs=bs, amp=amp, peft=peft)
        if hasattr(model, "device"):
            model.device = dev
        move_deep(module, dev)
        from .peft import make_trainable
        trained = make_trainable(model, module, peft, r=cfg.lora_r, targets=cfg.lora_targets)
        opt = torch.optim.AdamW([p for p in module.parameters() if p.requires_grad],
                                lr=cfg.eff_lr(), weight_decay=cfg.weight_decay)
        steps = cfg.eff_steps(len(windows))
        if getattr(cfg, "reward_steps", 0) and cfg.steps is None:
            # the matrix cap: it checks that reward post-training *runs*, in seconds.
            # An explicit --steps request is authoritative — silently trimming a
            # 48-step run to 6 is how a demonstrator "did not improve"
            steps = min(steps, int(cfg.reward_steps))
        self.n_candidates = max(2, min(self.n_candidates, 8 if dev == "cuda" else 4))
        log = StepLog(cfg.log_every)
        rng = np.random.RandomState(cfg.seed)
        ref = {n: p.detach().clone() for n, p in module.named_parameters() if p.requires_grad} \
            if self.kl > 0 else {}
        rhist = []
        tried_fallbacks: list[str] = []

        module.train()
        for step in range(1, steps + 1):
            cs, ys = _pick(windows, cfg.bs, rng, cfg.ctx)
            cs_t = torch.tensor(cs, dtype=torch.float32, device=dev)
            ys_t = torch.tensor(ys, dtype=torch.float32, device=dev)
            anchor = cs_t[:, -1][:, None]          # (B,1): broadcast over the horizon
            try:
                if mode == "grpo":
                    loss, rew = self._grpo_step(cs, ys, dev)
                elif mode == "reward_descent_point":
                    loss, rew = self._point_step(cs, ys_t, anchor)
                elif mode == "reward_descent":
                    from .losses import reward_from_quantiles
                    q = model.forward_quantiles(cs, cfg.horizon)
                    lv = getattr(model, "levels", LEVELS)[: q.shape[-1]]
                    rew_t = reward_from_quantiles(q, ys_t, lv, self.spec.w_pinball,
                                                  self.spec.w_dir, self.spec.w_cross, anchor)
                    loss, rew = -rew_t, float(rew_t.item())
                else:                                            # rft
                    loss, rew = self._rft_step(cs, ys, dev, rng)
            except CapabilityError:
                raise
            except Exception as e:                                # noqa: BLE001
                import os as _os
                if _os.environ.get("TSFMTRL_DEBUG"):
                    # the ladder deliberately retries with another mode, so the reason
                    # that ends up in the matrix is the last attempt's, not the first
                    # one's — with the switch on, every attempt prints its own trace
                    import sys, traceback
                    traceback.print_exc()
                    sys.stderr.write(f"[TSFMTRL_DEBUG] reward mode {mode} failed\n")
                # one graceful degradation ladder: reward_descent (needs a
                # differentiable quantile head) -> rft (works on the inference path),
                # then shrink the batch; anything else is reported, not swallowed
                if mode == "reward_descent" and _has_point_head(model):
                    # a quantile head that turned out to have nothing differentiable
                    # behind it (no levels declared): the point head is the honest
                    # gradient path, not a rejection-sampling detour
                    mode, tried_fallbacks = "reward_descent_point", tried_fallbacks + \
                        ["reward_descent->reward_descent_point"]
                    loss, rew = self._point_step(cs, ys, dev)
                elif mode in ("reward_descent", "reward_descent_point"):
                    mode, tried_fallbacks = "rft", tried_fallbacks + [f"{mode}->rft"]
                    loss, rew = self._rft_step(cs, ys, dev, rng)
                elif cfg.bs > 1:
                    cfg.bs = 1
                    tried_fallbacks = tried_fallbacks + ["bs->1"]
                    loss, rew = (self._rft_step(cs, ys, dev, rng) if mode == "rft"
                                 else self._grpo_step(cs, ys, dev))
                    from .forecast import _free
                    _free()
                else:
                    raise
            if loss is None:
                raise CapabilityError(key, "finetune", "reward step returned no loss")
            if self.kl > 0 and ref:
                kl = 0.0
                for n, p in module.named_parameters():
                    if n in ref and p.requires_grad:
                        kl = kl + (p - ref[n]).pow(2).mean()
                loss = loss + self.kl * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in module.parameters() if p.requires_grad],
                                           cfg.grad_clip)
            opt.step()
            rhist.append([step, round(float(rew), 5)])
            log.step(step, float(loss.item()))
            if cfg.max_seconds and time.time() - t0 > cfg.max_seconds:
                break
        module.eval()
        meta = {"model": key, "mode": mode, "reward": self.spec.to_dict(),
                "n_candidates": self.n_candidates, "kl": self.kl, "dataset": info.get("series"),
                "split_end": info.get("split_end"), "steps": steps, "budget": cfg.budget,
                "smoke": cfg.smoke, "trainable_parameters": count_trainable(module),
                "trained_parameters": trained[:40], "reward_first": rhist[0][1] if rhist else None,
                "reward_last": rhist[-1][1] if rhist else None,
                "loss_first": log.first, "loss_last": log.last,
                "fallbacks": fallbacks + tried_fallbacks}
        ck = ""
        if not cfg.smoke:
            from .checkpoints import save_ckpt
            ck = save_ckpt(cfg.ckpt_path(f"{key}__{mode}"), module, meta, cfg.to_dict(), log.steps)
        return TrainResult(model=key, status="smoke" if cfg.smoke else "ok", steps=steps,
                           first_loss=log.first, final_loss=log.last, checkpoint=ck,
                           trainable_parameters=count_trainable(module), n_windows=len(windows),
                           seconds=time.time() - t0,
                           extra={"mode": mode, "fallbacks": fallbacks + tried_fallbacks,
                                  "reward_first_last":
                                  [rhist[0][1], rhist[-1][1]] if rhist else [],
                                  **_trend(rhist)})

    # ------------------------------------------------------------------ steps
    def _point_step(self, cs, ys_t, anchor):
        """Reward descent through a *mean* head: scale-free accuracy + direction, no
        interval terms. Used for point-only TSFMs (MOMENT forecasting head, small TTM),
        where there is no distribution to score but the forecast is differentiable."""
        import torch
        pred = self.model.forward_prediction(list(cs), self.cfg.horizon).float()
        if pred.shape[1] != ys_t.shape[1]:
            pred = pred[:, :ys_t.shape[1]]
        scale = (ys_t - anchor).abs().mean(dim=1, keepdim=True).clamp_min(1e-6)
        acc = -(pred - ys_t).abs() / scale
        dirn = torch.sign(pred - anchor) * torch.sign(ys_t - anchor)
        rew = self.spec.w_pinball * acc + self.spec.w_dir * dirn
        rew = rew.clamp(-self.spec.clip, self.spec.clip)
        return -rew.mean(), float(rew.mean().item())

    def _rft_step(self, cs, ys, dev, rng):
        """Best-of-N: draw K futures from the model's **own predictive distribution**
        (inverse-CDF on its quantile grid, or its real sampled paths when it has
        them), reward every candidate, then take a supervised step toward the
        group-best through the family's real loss path (``training_step``).

        Using the inference path for candidates and the family loss for the step
        means the reward trainer works for every trainable model, and optimises the
        same objective the forecaster is scored with."""
        from ..model.base import LEVELS
        model = self.model
        K = self.n_candidates
        pred = model.predict(list(cs), self.cfg.horizon)
        H = self.cfg.horizon
        if getattr(pred, "samples", None) is not None:
            cands = np.asarray(pred.samples, float)[:, :K, :H]
            if cands.shape[1] < K:                     # pad with quantile draws
                q = np.asarray(pred.quantiles, float) if pred.quantiles is not None else None
                extra = _draw_from_grid(q, getattr(pred, "levels", LEVELS), K - cands.shape[1], rng)
                cands = np.concatenate([cands, extra], axis=1)
        elif pred.quantiles is not None:
            cands = _draw_from_grid(np.asarray(pred.quantiles, float),
                                    getattr(pred, "levels", LEVELS), K, rng)
        else:
            # A point-only forecaster gives best-of-N nothing to rank: K "candidates"
            # would be identical up to float noise, and the resulting step is a no-op
            # dressed up as post-training. Say so instead.
            raise CapabilityError(getattr(model, "key", "?"), "rft",
                                  "model exposes no quantiles or sample paths, so there is no "
                                  "predictive distribution to draw candidates from "
                                  "(wire the vendor's sampler into the adapter to enable RFT)")
        r = reward_np(cands, ys, cs[:, -1][:, None], self.spec)          # (B,K)
        adv = r - r.mean(axis=1, keepdims=True)
        best = np.argmax(adv, axis=1)
        target = cands[np.arange(len(cs)), best]
        loss = model.training_step(cs, target.astype(np.float32), self.cfg.horizon)
        return loss, float(r.max(axis=1).mean())

    def _grpo_step(self, cs, ys, dev):
        """Group-relative policy gradient over the model's own sampled paths."""
        import torch
        model = self.model
        K = self.n_candidates
        paths = model.sample_paths(cs, self.cfg.horizon, n=K)           # (B,K,H)
        if not np.all(np.isfinite(paths)):
            raise CapabilityError(getattr(model, "key", "?"), "paths", "non-finite samples")
        B = paths.shape[0]
        r = np.zeros((B, K))
        for b in range(B):
            # scale-free point accuracy + directional term, exactly the language of
            # the default reward spec (so the two modes are comparable)
            scale = max(1e-9, float(np.std(np.r_[cs[b, -12:], ys[b]])))
            anchor = float(cs[b, -1])
            for k in range(K):
                acc = -float(np.mean(np.abs(paths[b, k] - ys[b]))) / scale
                dirn = float(np.mean(np.sign(paths[b, k] - anchor) == np.sign(ys[b] - anchor)))
                r[b, k] = self.spec.w_dir * dirn - self.spec.w_pinball * acc
        adv = (r - r.mean(axis=1, keepdims=True)) / (r.std(axis=1, keepdims=True) + 1e-6)
        loss = None
        for b in range(B):
            k = int(np.argmax(adv[b]))
            lp = model.path_logprob(cs[b:b + 1], paths[b, k:k + 1])[0]   # scalar tensor
            term = -torch.as_tensor(float(adv[b, k]), dtype=lp.dtype, device=lp.device) * lp
            loss = term if loss is None else loss + term
        return (loss / max(1, B)), float(r.mean())

    def _dir_term(self, q, y, anchor):
        import torch
        lv = getattr(self.model, "levels", LEVELS)
        med = int(np.argmin(np.abs(np.asarray(lv, float) - 0.5)))
        return (torch.sign(q[..., med] - anchor) * torch.sign(y - anchor)).mean()


def _trend(rhist: list) -> dict:
    """Reward over time, stated the way it can actually be read.

    A reward curve on a handful of windows is noisy batch-to-batch: first step vs
    last step says mostly nothing (a 96-step GRPO run can read 1.73 -> 1.60 while
    the *late* steps sit clearly above the *early* ones). The report therefore gives
    quartile means, and "improved" compares the last quartile with the first — the
    honest version of "did post-training help".
    """
    if len(rhist) < 4:
        return {}
    vals = [v for _, v in rhist]
    q = max(1, len(vals) // 4)
    early = sum(vals[:q]) / q
    late = sum(vals[-q:]) / q
    return {"reward_early_mean": round(early, 5), "reward_late_mean": round(late, 5),
            "improved": late > early}


def _has_point_head(model) -> bool:
    """True when the *adapter* implements a differentiable point head. Checked on the
    class (not by calling it): probing would build a whole model graph, and the base
    class method always raises NotImplementedError by design."""
    from ..model.base import TSFMModel
    fn = getattr(type(model), "forward_prediction", None)
    return fn is not None and fn is not TSFMModel.forward_prediction


def _draw_from_grid(q, levels, K, rng):
    """Inverse-CDF draws (piecewise-linear on the model's quantile grid). q is
    (B,H,nq) -> (B,K,H). A cheap way to get on-manifold candidates from a quantile
    head that cannot sample paths."""
    q = np.asarray(q, dtype=float)
    lv = np.asarray(levels, dtype=float).reshape(-1)
    B, H, nq = q.shape
    if lv.size != nq:
        # the grid the caller named is not the grid the head emitted (a Pred built
        # without levels, or a model whose knots were remapped). np.interp would raise
        # "fp and xp are not of the same length"; a uniform grid over what is actually
        # there is the honest reading for candidate *sampling*
        lv = (np.arange(1, nq + 1) / (nq + 1))
    ext = np.r_[0.0, lv, 1.0]
    qs = np.concatenate([q[..., :1], q, q[..., -1:]], axis=-1)
    out = np.empty((B, K, H))
    for b in range(B):
        for h in range(H):
            out[b, :, h] = np.interp(rng.rand(K), ext, qs[b, h])
    return out


def reward_np(cands, truth, anchor, spec: RewardSpec):
    """Numpy reward for candidates (B,K,H) vs truth (B,H): scale-free accuracy +
    directional term (see :class:`RewardSpec`)."""
    B, K, H = cands.shape
    scale = np.array([max(1e-9, float(np.std(np.r_[anchor[b], truth[b]]))) for b in range(B)])
    acc = -np.abs(cands - truth[:, None, :]).mean(axis=2) / scale[:, None]
    a = np.asarray(anchor).reshape(-1, 1, 1)
    dirn = (np.sign(cands - a) == np.sign(truth[:, None, :] - a)).mean(axis=2)
    r = spec.w_pinball * acc + spec.w_dir * dirn.astype(float)
    return np.clip(r, -spec.clip, spec.clip)


def _pick(windows, bs, rng, ctx):
    if len(windows) == 0:
        from ..errors import DataError
        raise DataError("no windows to sample from (check --ctx/--horizon against the "
                        "series length, or pass more windows)")
    idx = rng.choice(len(windows), size=max(1, min(bs, len(windows))), replace=False)
    cs = np.stack([np.asarray(windows[i][0], float)[-ctx:] for i in idx])
    ys = np.stack([np.asarray(windows[i][1], float) for i in idx])
    return cs, ys
