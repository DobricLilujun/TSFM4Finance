"""``ForecastTrainer`` — one loop, per-family step functions.

Why the split is shaped this way: the *loop* (minibatches, AMP, grad clip, warmup,
budgets, checkpointing, OOM fallbacks) is identical for every model, but the *step*
is not — scaling, head and loss differ per family, and a single generic step would
silently change what "fine-tuned X" means. So the loop is generic and the step
lives in the adapter (``model.training_step``), except for families whose vendor
owns the loop (Chronos-2 ``fit``, Kronos ``finetune_csv``), which implement
``model.vendor_fit`` instead.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..data.series import SeriesDataset, train_windows
from ..errors import CapabilityError, DataError
from .checkpoints import save_ckpt
from .config import StepLog, TrainResult, TrainingConfig
from .peft import (count_trainable, make_trainable, maybe_scheduler, merge_lora,
                   move_deep,
                     optimiser)

# memory fallbacks, in order: what to try when a step dies on VRAM
_LADDER = [("keep", None), ("amp", "bf16"), ("amp", "fp16"), ("amp", "fp32"),
           ("lr", 0.1), ("ctx", 2), ("bs", 1),
           ("device", "cpu")]


class ForecastTrainer:
    def __init__(self, model, cfg: TrainingConfig | None = None, **cfg_kw):
        self.model = model
        self.cfg = cfg or TrainingConfig(**cfg_kw)

    # ------------------------------------------------------------------ data
    def _windows(self, data):
        cfg = self.cfg
        if isinstance(data, SeriesDataset):
            return train_windows(data, cfg.horizon, cfg.ctx, split=cfg.split,
                                 max_windows=cfg.eff_n_windows(), seed=cfg.seed)
        if isinstance(data, (str, Path)):
            return self._windows(SeriesDataset.load(data))
        if isinstance(data, dict) and "windows" in data:
            return data["windows"], data.get("info", {})
        if isinstance(data, (list, tuple)) and data and isinstance(data[0], (tuple, list)) \
                and len(data[0]) == 2:
            return list(data), {"series": "in-memory", "n_windows": len(data)}
        # raw series
        arr = np.asarray(data, dtype=float)
        from ..data.series import windows
        w = windows(arr, cfg.horizon, cfg.ctx, max_windows=cfg.eff_n_windows(), seed=cfg.seed)
        if not w:
            raise DataError(f"no windows: need >= {cfg.ctx + cfg.horizon} observations")
        return w, {"series": "array", "n_windows": len(w)}

    # ------------------------------------------------------------------ fit
    def fit(self, data, *, resume_profile: dict | None = None) -> TrainResult:
        cfg = self.cfg
        windows, info = self._windows(data)
        model_key = getattr(self.model, "key", "model")
        t0 = time.time()

        # ---- vendor-owned loops -------------------------------------------
        if getattr(self.model, "loss_kind", None) == "vendor":
            try:
                out = self.model.vendor_fit(windows, cfg)
            except CapabilityError as e:
                return TrainResult(model=model_key, status="skipped", reason=str(e),
                                   n_windows=len(windows))
            return self._result_from_vendor(model_key, out, windows, info, t0)

        # ---- generic loop with the memory ladder ---------------------------
        ladder = [(a, b) for a, b in _LADDER]
        ctx, bs, amp, device = cfg.ctx, cfg.bs, cfg.amp, cfg.device
        lr_mult = 1.0
        last_err = None
        for j, (kind, val) in enumerate(ladder):
            if kind == "amp":
                amp = val
            elif kind == "lr":
                # a big model at the fine-tuning rate can produce its first non-finite
                # loss before the optimiser ever sees a gradient worth keeping; the
                # ladder's answer is a smaller step, not a verdict on the family
                lr_mult = float(val)
            elif kind == "ctx":
                ctx = max(32, ctx // 2)
            elif kind == "bs":
                bs = 1
            elif kind == "device":
                device = "cpu"
            try:
                return self._run(windows, info, model_key, ctx=ctx, bs=bs, amp=amp,
                                 device=device, tried_fallbacks=ladder[:j], t0=t0,
                                 lr_mult=lr_mult)
            except CapabilityError as e:
                return TrainResult(model=model_key, status="skipped", reason=str(e),
                                   n_windows=len(windows))
            except Exception as e:                                      # noqa: BLE001
                last_err = e
                if not (_is_oom(e) or _is_env(e) or _is_unstable(e)):
                    return TrainResult(model=model_key, status="failed",
                                       reason=f"{type(e).__name__}: {e}"[:400],
                                       n_windows=len(windows),
                                       seconds=time.time() - t0)
                _free()
                continue
        return TrainResult(model=model_key, status="failed",
                           reason=f"out of memory even on CPU ({last_err})"[:400],
                           n_windows=len(windows), seconds=time.time() - t0)

    # ------------------------------------------------------------------ one attempt
    def _run(self, windows, info, model_key, *, ctx, bs, amp, device,
             tried_fallbacks, t0, lr_mult: float = 1.0) -> TrainResult:
        import torch
        cfg = self.cfg
        model = self.model
        model.prepare_training(ctx, cfg.horizon)      # shape-dependent graphs (MOMENT)
        module = model.trainable()
        if module is None:
            raise CapabilityError(model_key, "finetune", "adapter exposes no trainable module")
        if device in ("", "auto") or device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # the adapter must place *its* tensors on the same device the ladder chose
        if getattr(model, "device", None) != device and hasattr(model, "device"):
            model.device = device
        moved = move_deep(module, device)
        if moved:
            self._notes = getattr(self, "_notes", []) + [f"moved {moved} non-registered "
                                                         f"tensor(s) to {device}"]

        trained = make_trainable(model, module, cfg.peft, r=cfg.lora_r,
                                 targets=cfg.lora_targets)
        n_train = count_trainable(module)

        steps = cfg.eff_steps(len(windows))
        opt = optimiser(module, cfg.eff_lr() * lr_mult, cfg.weight_decay)
        sched = maybe_scheduler(opt, cfg.warmup, steps)
        rng = np.random.RandomState(cfg.seed)
        log = StepLog(cfg.log_every)

        module.train()
        for step in range(1, steps + 1):
            cs, ys = _batch(windows, bs, rng, ctx)
            from ..model.base import autocast
            with autocast(device, amp):
                loss = model.training_step(cs, ys, cfg.horizon)
            if loss is None:
                raise CapabilityError(model_key, "finetune", "training_step returned None")
            if not torch.is_tensor(loss):
                loss = torch.as_tensor(float(loss), device=device)
            if loss.ndim:
                # some vendors hand back a per-window loss vector (Kairos does). Reducing
                # it to its mean is the only step that can be optimised; doing the truth
                # test on the vector instead is what used to raise the "Boolean value of
                # Tensor" error that got filed as a model limitation.
                self._notes = getattr(self, "_notes", []) + \
                    [f"per-element loss ({tuple(loss.shape)}) reduced to its mean"]
                loss = loss.mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_([p for p in module.parameters() if p.requires_grad],
                                               cfg.grad_clip)
            opt.step()
            if sched:
                sched.step()
            log.step(step, float(loss.item()))
            if cfg.max_seconds and time.time() - t0 > cfg.max_seconds:
                break
        module.eval()
        if cfg.peft == "lora":
            merge_lora(module)

        meta = {"model": model_key, "family": getattr(model, "family", "?"),
                "dataset": info.get("series"), "budget": cfg.budget, "smoke": cfg.smoke,
                "split": cfg.split, "split_end": info.get("split_end"),
                "split_end_index": info.get("split_end_index"),
                "n_windows": len(windows), "steps": steps, "ctx": ctx, "horizon": cfg.horizon,
                "trainable_parameters": n_train, "trained_parameters": trained[:40],
                "peft": cfg.peft, "seed": cfg.seed,
                "profile": {"device": device, "amp": amp, "bs": bs, "ctx": ctx,
                            "fallbacks": [f"{k}:{v}" for k, v in tried_fallbacks if k != "keep"]},
                "loss_first": log.first, "loss_last": log.last,
                "seconds": round(time.time() - t0, 1)}
        # a 2-step smoke checkpoint is not an artifact: writing one per model per
        # matrix run put 7.6 GB of MOMENT and TimesFM copies on disk and proved nothing.
        # The row keeps the loss curve and the trainable-parameter count; the weights
        # are regenerable from the same seed.
        ck = "" if cfg.smoke else save_ckpt(cfg.ckpt_path(model_key), module, meta,
                                            cfg.to_dict(), log.steps)
        _free()
        return TrainResult(model=model_key, status="smoke" if cfg.smoke else "ok",
                           steps=steps, first_loss=log.first, final_loss=log.last,
                           checkpoint=str(ck), trainable_parameters=n_train,
                           trained_parameters=trained[:40], n_windows=len(windows),
                           seconds=time.time() - t0, device=device,
                           extra={"profile": meta["profile"], "split_end": info.get("split_end")})

    def _result_from_vendor(self, model_key, out, windows, info, t0) -> TrainResult:
        if isinstance(out, TrainResult):
            return out
        if isinstance(out, dict):
            ck = out.get("checkpoint", "")
            if not ck and out.get("state_dict") is not None:
                ck = save_ckpt(self.cfg.ckpt_path(model_key), out["state_dict"],
                               {**out.get("meta", {}), "model": model_key,
                                "budget": self.cfg.budget, "smoke": self.cfg.smoke,
                                "split_end": info.get("split_end"),
                                "vendor_fit": True},
                               self.cfg.to_dict(), out.get("loss"))
            # a vendor loop that declined to run (e.g. Kronos' finetune_csv unless
            # run_training=True) must not be reported as a fine-tune
            st = out.get("status")
            if st in ("prepared", "failed", "skipped"):
                return TrainResult(model=model_key, status=st,
                                   reason=out.get("reason", st),
                                   steps=int(out.get("steps", 0)),
                                   seconds=time.time() - t0, n_windows=len(windows),
                                   extra={"vendor": out})
            return TrainResult(model=model_key, status="smoke" if self.cfg.smoke else "ok",
                               steps=int(out.get("steps", 0)),
                               first_loss=float(out.get("first_loss", float("nan"))),
                               final_loss=float(out.get("final_loss", float("nan"))),
                               checkpoint=str(ck),
                               trainable_parameters=int(out.get("trainable_parameters", 0)),
                               n_windows=len(windows), seconds=time.time() - t0,
                               extra={"vendor": True, **{k: v for k, v in out.items()
                                                          if k not in ("state_dict",)}})
        raise RuntimeError(f"vendor_fit returned {type(out).__name__}")


def _batch(windows, bs, rng, ctx):
    idx = rng.choice(len(windows), size=min(bs, len(windows)), replace=False)
    cs = np.stack([np.asarray(windows[i][0], dtype=float)[-ctx:] for i in idx])
    ys = np.stack([np.asarray(windows[i][1], dtype=float) for i in idx])
    return cs, ys


def _is_oom(e: Exception) -> bool:
    s = f"{type(e).__name__} {e}".lower()
    return "out of memory" in s or "cuda out of memory" in s or "insufficient memory" in s \
        or "cannot allocate" in s or "cublas_status_alloc" in s


def _is_unstable(e) -> bool:
    """A non-finite loss is a numerical condition of this profile, not a property of
    the family: fp32 or a tenth of the step usually trains the same window fine."""
    m = str(e).lower()
    return "non-finite" in m or "nan" in m or "inf" in m


def _is_env(e: Exception) -> bool:
    """A CUDA *toolchain* failure (missing nvrtc builtins, an arch mismatch, an
    unimplemented kernel), not a data or model failure. These are recoverable by
    moving the run to CPU — the matrix should report a slow fine-tune with a
    fallback profile, not a phantom "model cannot be fine-tuned"."""
    s = f"{type(e).__name__} {e}".lower()
    return any(k in s for k in ("libnvrtc", "nvrtc", "not implemented", "invalid device function",
                                "unsupported cast", "no kernel image", "cuda error",
                                "operation not supported"))


def _free() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                          # noqa: BLE001
        pass
