"""Engine contract for the multi-model sweep.

One engine = one (family, checkpoint) pair.  Every engine speaks the same
language so the protocol, the metrics and the trading layer stay model-agnostic:

    predict(contexts, horizon) -> Pred(point (N,H), quantiles (N,H,9) | None, ...)
    encode(contexts)           -> (N, D) embeddings      [cap: "embed"]
    classify(...) / anomaly(...)                            [cap: "clf_head"/"anomaly_head"]
    trainable()                -> (module, loss_fn)        [cap: "finetune"]

Quantile convention: engines may emit any quantile grid; ``to_levels`` maps it to
the benchmark's fixed LEVELS by linear interpolation, so CRPS/WQL/VaR/coverage are
computed on identical levels for every model (a fair comparison requires this).
"""
from __future__ import annotations

import os as _os
import sys as _sys

# Model packages compile CUDA/C++ helpers on demand (flashrnn needs ``ninja``).
# Adapters are usually launched as ``<venv>/bin/python -m ...`` without activating
# the venv, so put that venv's bin directory on PATH.
_bindir = _os.path.dirname(_sys.executable)
if _bindir and _bindir not in _os.environ.get("PATH", "").split(_os.pathsep):
    _os.environ["PATH"] = _bindir + _os.pathsep + _os.environ.get("PATH", "")

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MAX_CONTEXT_CAP = 512          # benchmark cap; models with smaller windows cap lower


@dataclass
class Pred:
    point: np.ndarray                       # (N, H)
    quantiles: np.ndarray | None = None     # (N, H, 9) at LEVELS, or None
    samples: np.ndarray | None = None       # (N, S, H) if the model samples paths
    levels: list[float] = field(default_factory=lambda: list(LEVELS))
    meta: dict = field(default_factory=dict)


def to_levels(q: np.ndarray, src_levels: Sequence[float]) -> np.ndarray:
    """Interpolate an arbitrary quantile grid onto LEVELS, per (row, horizon)."""
    q = np.asarray(q, dtype=float)
    src = np.asarray(src_levels, dtype=float)
    if q.ndim == 2:                                   # (N,H) single quantile
        q = q[..., None]
    if q.shape[-1] != src.size:
        raise ValueError(f"quantile axis {q.shape[-1]} != src levels {src.size}")
    if np.array_equal(np.round(src, 6), np.round(np.asarray(LEVELS, float), 6)):
        return q
    out = np.empty(q.shape[:-1] + (len(LEVELS),))
    it = np.nditer(q[..., 0], flags=["multi_index"])
    for _ in it:
        mi = it.multi_index
        row = q[mi]
        if not np.isfinite(row).all():
            out[mi] = np.nan
            continue
        # quantile functions must be monotone for interpolation to be meaningful
        order = np.argsort(src)
        xs, ys = src[order], row[order]
        ys = np.maximum.accumulate(ys)
        out[mi] = np.interp(np.asarray(LEVELS, float), xs, ys)
    return out


def quantiles_from_samples(samples: np.ndarray) -> np.ndarray:
    s = np.asarray(samples, dtype=float)
    if s.ndim == 2:
        s = s[:, None, :]
    return np.quantile(s, LEVELS, axis=1).transpose(1, 0, 2)


def median_of(quantiles: np.ndarray) -> np.ndarray:
    return np.asarray(quantiles, float)[..., 4]


def cap_context(contexts: Sequence[np.ndarray], cap: int) -> list[np.ndarray]:
    return [np.asarray(c, dtype=float)[-cap:] for c in contexts]


class Engine:
    # ---- adapted (fine-tuned) checkpoints -------------------------------------
    # Family trainers save {"model": state_dict, "meta": {...}} under
    # reports/ft_ckpts/<model>__<series>__<budget>.pt.  An adapted engine is built
    # from the BASE weights and then gets that state dict loaded into its module,
    # so every family shares one contract instead of bespoke loading code.
    _FT_ATTRS = ("module", "model", "net", "predictor", "pipe")

    def _ft_target(self):
        for attr in self._FT_ATTRS:
            m = getattr(self, attr, None)
            if m is None:
                continue
            inner = getattr(m, "model", m)      # Chronos: pipe.model
            if hasattr(inner, "load_state_dict"):
                return inner, f"{attr}"
        return None, None

    def load_finetuned(self, path: str) -> dict:
        import torch
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
        state = obj.get("model", obj) if isinstance(obj, dict) else obj
        target, via = self._ft_target()
        if target is None or not hasattr(state, "keys"):
            raise RuntimeError(f"{type(self).__name__}: cannot apply fine-tuned "
                               f"checkpoint {path} (no torch module / no state dict)")
        full = dict(target.state_dict())
        missing, unexpected = target.load_state_dict(state, strict=False)
        if len(missing) == len(full):
            raise RuntimeError(f"{type(self).__name__}: fine-tuned checkpoint keys do "
                               f"not match this module via '{via}' "
                               f"(missing={len(missing)} unexpected={len(unexpected)})")
        meta = obj.get("meta", {}) if isinstance(obj, dict) else {}
        self.ft_meta = meta
        self.ft_load = {"via": via, "missing": len(missing), "unexpected": len(unexpected),
                        "checkpoint": str(path)}
        return self.ft_load
    """Base class. Subclasses set ``key``, ``caps`` and implement ``predict``."""
    key: str = "engine"
    family: str = "engine"
    caps: set[str] = set()
    max_context: int = MAX_CONTEXT_CAP
    needs: tuple[str, ...] = ("close",)          # dataset columns required
    ctx_step: int = 1                           # context must be a multiple of this
    max_horizon: int = 512

    def __init__(self, *, device: str = "cuda", amp: str = "fp32", batch: int = 16,
                 ctx_cap: int | None = None, seed: int = 0, **_ignored):
        self.device = device
        self.amp = amp
        self.batch = max(1, int(batch))
        if ctx_cap:
            self.max_context = min(self.max_context, int(ctx_cap))
        self.seed = seed

    # ------------------------------------------------------------------ caps
    def can(self, cap: str) -> bool:
        return cap in self.caps

    def trim(self, contexts: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Cut contexts to ``max_context`` and (for patch-based models) to a
        multiple of the patch length, keeping the most recent observations."""
        cs = cap_context(contexts, self.max_context)
        if self.ctx_step > 1:
            out = []
            for c in cs:
                keep = (len(c) // self.ctx_step) * self.ctx_step or self.ctx_step
                out.append(c[-keep:])
            cs = out
        return cs

    # ------------------------------------------------------------------ core
    def predict(self, contexts: Sequence[np.ndarray], horizon: int, *,
                panels: Sequence[object] | None = None) -> Pred:
        raise NotImplementedError

    # ------------------------------------------------------------------ extras
    def encode(self, contexts: Sequence[np.ndarray]) -> np.ndarray:
        raise NotImplementedError(f"{self.key}: no embedding support")

    def classify(self, contexts: Sequence[np.ndarray], labels: np.ndarray,
                 kind: str = "logistic") -> dict:
        raise NotImplementedError(f"{self.key}: no native classification head")

    def anomaly_scores(self, contexts: Sequence[np.ndarray]) -> np.ndarray:
        raise NotImplementedError(f"{self.key}: no native anomaly head")

    def trainable(self):
        """Return (module, loss_fn) for the model-specific fine-tuning trainer.

        ``loss_fn(out, y_raw, params)`` must be differentiable and scale-aware:
        financial series differ by orders of magnitude, so every trainer here
        scores in a normalized space (pinball on per-window scaled values).
        """
        raise NotImplementedError(f"{self.key}: no fine-tuning trainer")

    def close(self) -> None:
        pass


def env_note(msg: str) -> str:                              # pragma: no cover
    return f"[env] {msg}"
