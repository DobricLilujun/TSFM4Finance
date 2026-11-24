"""Prediction contract and the model base class.

Every adapter in :mod:`tsfm_trl.model.adapters` speaks the same language, so the
trainers, the classifier routes and the evaluator never branch on the model:

    predict(contexts, horizon) -> Pred(point (N,H), quantiles (N,H,nq) | None,
                                       samples (N,S,H) | None)
    encode(contexts)           -> (N, D) embeddings                  [cap: embed]
    classifier_features(...)   -> differentiable (B, D) trunk tensor [cap: clf_head]
    trainable()                -> nn.Module to optimise              [cap: finetune]
    training_step(ctx, truth)  -> differentiable scalar loss         [cap: finetune]

Quantile convention: a model may emit *any* grid; ``to_levels`` maps it onto the
library's fixed ``LEVELS`` by monotone interpolation, so CRPS/WQL/pinball/VaR are
computed on identical levels for every model.  Without that, a "model comparison"
is just two different measurements.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..errors import CapabilityError, UnavailableError

__all__ = ["Pred", "LEVELS", "MAX_CONTEXT", "CAPS", "to_levels", "levels_from_count",
           "quantiles_from_samples", "TSFMModel", "autocast",
           "CapabilityError", "UnavailableError"]

LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MAX_CONTEXT = 512

# capability vocabulary used by the registry, the doctor matrix and the trainers
CAPS = frozenset({"forecast", "quantiles", "embed", "finetune", "clf_head", "anomaly_head",
                  "multivariate", "covariates", "panel_in", "paths"})


@dataclass
class Pred:
    """One batch of forecasts. ``point`` is (N,H); ``quantiles`` is (N,H,9) on
    ``levels``; ``samples`` is (N,S,H) when the model samples paths."""

    point: np.ndarray
    quantiles: np.ndarray | None = None
    samples: np.ndarray | None = None
    levels: list[float] = field(default_factory=lambda: list(LEVELS))
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # the contract says every interval says how it was made. Adapters that go to the
        # trouble of naming their method (empirical from k sampled paths, temperature AR
        # decode, median-rollback rollout) are left alone; the rest get the plain truth
        if self.quantiles is not None and "how" not in self.meta:
            self.meta["how"] = ("native quantile head" if self.samples is None
                                else "empirical from sampled paths")

    def median(self) -> np.ndarray:
        if self.quantiles is None:
            return self.point
        i = int(np.argmin(np.abs(np.asarray(self.levels) - 0.5)))
        return np.asarray(self.quantiles)[..., i]

    def quantile(self, level: float) -> np.ndarray:
        if self.quantiles is None:
            raise ValueError("model returned no quantiles")
        i = int(np.argmin(np.abs(np.asarray(self.levels) - float(level))))
        return np.asarray(self.quantiles)[..., i]


def to_numpy(v) -> np.ndarray:
    """Any vendor tensor (torch on GPU, jax, pandas) to a float numpy array.

    Several vendor forecast objects keep their outputs on the GPU, and ``np.asarray``
    on a CUDA tensor raises ``TypeError: can't convert cuda:0 device type tensor to
    numpy`` — which a matrix run would otherwise file as a failure *of the model*.
    """
    if v is None:
        return None
    if hasattr(v, "detach"):                        # torch (any device, any grad state)
        return v.detach().float().cpu().numpy()
    if hasattr(v, "device") and hasattr(v, "numpy"):  # jax / tf tensor
        try:
            return np.asarray(v, dtype=np.float64)
        except (TypeError, ValueError):
            return np.asarray(v.cpu().numpy(), dtype=np.float64)
    return np.asarray(v, dtype=np.float64)


def to_levels(q: np.ndarray, src_levels: Sequence[float]) -> np.ndarray:
    """Interpolate an arbitrary quantile grid (..., nq) onto ``LEVELS``.

    The interpolation is done on the sorted source grid and the result is forced
    monotone, because a handful of released heads are numerically non-monotone at
    the tails and a crossing quantile function breaks CRPS.
    """
    q = np.asarray(q, dtype=float)
    src = np.asarray(src_levels, dtype=float)
    if q.ndim == 2:                                   # (N,H) -> one quantile
        q = q[..., None]
    if q.shape[-1] != src.size:
        raise ValueError(f"quantile axis {q.shape[-1]} != src levels {src.size}")
    if np.array_equal(np.round(src, 6), np.round(np.asarray(LEVELS, float), 6)):
        return q
    order = np.argsort(src)
    xs = src[order]
    tgt = np.asarray(LEVELS, float)
    flat = q.reshape(-1, q.shape[-1])[:, order]
    res = np.empty((flat.shape[0], len(LEVELS)))
    for i, row in enumerate(flat):
        if not np.isfinite(row).all():
            res[i] = np.nan
            continue
        # maximum.accumulate enforces monotonicity: a handful of released heads are
        # numerically non-monotone at the tails, and crossing quantiles break CRPS
        res[i] = np.interp(tgt, xs, np.maximum.accumulate(row))
    return res.reshape(q.shape[:-1] + (len(LEVELS),))


def levels_from_count(nq: int) -> list[float]:
    """Best guess at the grid of a head that only tells us how many quantiles it
    emits (used by several vendor wrappers that hide their config)."""
    if nq == len(LEVELS):
        return list(LEVELS)
    return list(np.linspace(0.0, 1.0, nq + 2)[1:-1])


def quantiles_from_samples(samples: np.ndarray, levels: Sequence[float] = LEVELS) -> np.ndarray:
    s = np.asarray(samples, dtype=float)
    if s.ndim == 2:
        s = s[:, None, :]
    # np.quantile -> (nq, N, H); Pred wants (N, H, nq)
    return np.quantile(s, list(levels), axis=1).transpose(1, 2, 0)


class TSFMModel:
    """Base class: subclass sets ``key``, ``family``, ``caps`` and implements
    ``predict`` (plus whatever training hooks the family supports).

    Constructor knobs are the ones that actually differ between machines, so the
    same model string runs on a 6 GB laptop and a 48 GB card without code changes.
    """

    key: str = "model"
    family: str = "model"
    caps: frozenset[str] = frozenset({"forecast"})
    max_context: int = MAX_CONTEXT
    max_horizon: int = 512
    ctx_step: int = 1                        # patch-based models: context % ctx_step == 0
    needs: tuple[str, ...] = ("close",)      # dataset columns required
    # training contract: "quantile" (pinball on a quantile head), "token"
    # (cross-entropy over a discrete/continuous token head), "vendor" (the vendor
    # owns the loop, e.g. Chronos-2 ``fit``), or None (nothing to fine-tune)
    loss_kind: str | None = None
    head_prefixes: tuple[str, ...] = ()      # what ``peft="heads_only"`` unfreezes
    supports: dict = {}                      # free-form capability notes for the matrix

    def __init__(self, *, device: str = "cuda", amp: str = "fp32", batch: int = 16,
                 ctx_cap: int | None = None, seed: int = 0, source_id: str | None = None,
                 **_ignored):
        self.device = device
        self.amp = amp
        self.batch = max(1, int(batch))
        if ctx_cap:
            self.max_context = min(self.max_context, int(ctx_cap))
        self.seed = seed
        # the registry id (HF id) even when the weights resolved to a local directory:
        # loaders like TiRex validate the *id*, so the id has to survive resolution
        self.source_id = source_id
        self.ft_meta: dict = {}
        self.supports = dict(self.supports)   # per-instance; adapters annotate it

    # ------------------------------------------------------------------ caps
    def can(self, cap: str) -> bool:
        return cap in set(self.caps)

    def trim(self, contexts: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Cut contexts to ``max_context`` (and to a multiple of the patch length),
        always keeping the most recent observations.

        Accepts pandas objects: a ``Series`` (or single-column frame) is common when a
        caller forwards their own frame, and ``np.asarray`` on it would otherwise yield
        column *names* and fail with a confusing "could not convert string to float".
        """
        from ..data.series import as_1d
        out = []
        for c in contexts:
            c = as_1d(c)[-self.max_context:]
            if self.ctx_step > 1:
                keep = (len(c) // self.ctx_step) * self.ctx_step or self.ctx_step
                c = c[-keep:]
            out.append(c)
        return out

    # ------------------------------------------------------------------ core
    def predict(self, contexts: Sequence[np.ndarray], horizon: int, *,
                panels: Sequence[object] | None = None,
                future: Sequence[np.ndarray] | None = None) -> Pred:
        raise NotImplementedError

    # ------------------------------------------------------------------ extras
    def encode(self, contexts: Sequence[np.ndarray]) -> np.ndarray:
        """Frozen representation for probe-style classification. Not a capability
        claim: many models expose *some* representation, so the default derives
        one from the forecast (which every model in scope can produce)."""
        horizon = min(16, self.max_horizon)
        p = self.predict(contexts, horizon)
        n = len(list(contexts))
        feats = [np.asarray(p.point, float).reshape(n, -1)]
        if p.quantiles is not None:
            feats.append(np.asarray(p.quantiles, float).reshape(n, -1))
        if p.samples is not None:
            s = p.samples
            feats.append(np.stack([s.mean(axis=1), s.std(axis=1)], axis=1).reshape(len(contexts), -1))
        return np.concatenate(feats, axis=1)

    def classifier_features(self, contexts: Sequence[np.ndarray]) -> "np.ndarray":
        """Feature block handed to the classification head. Override to return a
        *differentiable* tensor when the trunk can be trained end-to-end."""
        return self.encode(contexts)

    def anomaly_scores(self, contexts: Sequence[np.ndarray]) -> np.ndarray:
        """Model-native anomaly score (higher = more anomalous)."""
        # generic fallback: forecast residual under the model's own uncertainty
        ctx = [np.asarray(c, float) for c in contexts]
        h = min(16, max(1, min(len(c) // 4 for c in ctx)))
        p = self.predict([c[:-h] for c in ctx], h)
        y = np.stack([c[-h:] for c in ctx])
        q = p.quantiles
        if q is None:
            # mean-only head (e.g. the small TTM releases): fall back to a residual
            # z-score against the context's own scale, and say what the number is
            self.supports["anomaly"] = "residual z-score (model emits no interval head)"
            scale = np.stack([np.std(c[-min(len(c), 60):]) for c in ctx])
            scale = np.where(scale > 0, scale, 1.0)
            # (B,) scale against a (B,h) residual: the scale needs its horizon axis or
            # numpy reads the trailing 4 as the horizon and refuses to broadcast
            return np.nanmean(np.abs(y - p.point) / scale[:, None], axis=1)
        lo, hi = p.quantile(0.1), p.quantile(0.9)
        if lo is None or hi is None or lo.shape != p.point.shape:
            # the model has an interval head but not at the band we asked for: score
            # against the context scale instead of producing a None-minus-float error
            self.supports["anomaly"] = "residual z-score (no 0.1/0.9 band in the grid)"
            scale = np.stack([np.std(c[-min(len(c), 60):]) for c in ctx])
            scale = np.where(scale > 0, scale, 1.0)
            return np.nanmean(np.abs(y - p.point) / scale[:, None], axis=1)
        scale = np.where(hi > lo, hi - lo, 1.0)
        return np.nanmean(np.abs(y - p.point) / scale, axis=1)

    # ------------------------------------------------------------------ training hooks
    def prepare_training(self, ctx: int, horizon: int) -> None:
        """Hook for families whose *module graph* depends on the shape being trained
        (MOMENT fixes ``seq_len``/``pred_len`` at init). The trainers call this before
        ``trainable()``; the default is a no-op."""

    def trainable(self):
        """The ``torch.nn.Module`` the trainer optimises. ``None`` = not trainable."""
        return None

    def forward_prediction(self, contexts, horizon):
        """Differentiable point forecast tensor (B,H) for reward descent on models whose
        head is a mean, not a distribution. Not every family can provide it; those that
        cannot say so and the reward trainer falls back to another mode."""
        raise NotImplementedError(f"{self.key}: no differentiable point head exposed")

    def training_step(self, contexts: np.ndarray, truth: np.ndarray, horizon: int):
        """Differentiable scalar loss for one minibatch (real units).

        Implemented per family because the families disagree about scaling, heads
        and which parameters may move — a single generic loop would silently
        redefine what "fine-tuned" means for each model.
        """
        raise CapabilityError(self.key, "finetune", "no training_step")

    def vendor_fit(self, windows, cfg):
        """Families whose vendor ships the loop (Chronos-2 ``fit``) return a
        state_dict here instead of using the generic trainer."""
        raise CapabilityError(self.key, "finetune", "no vendor_fit")

    def sample_paths(self, contexts: np.ndarray, horizon: int, n: int = 8):
        """(paths, log_prob) for reward/RL training on sampling models."""
        raise CapabilityError(self.key, "paths", "reward training needs sampled paths")

    # ---- fine-tuned checkpoint application (one contract for every family) ----
    _FT_ATTRS = ("module", "model", "net", "predictor", "pipe", "_mod", "_pipe")

    def _ft_target(self):
        for attr in self._FT_ATTRS:
            m = getattr(self, attr, None)
            if m is None:
                continue
            inner = getattr(m, "model", m)              # Chronos: pipe.model
            if hasattr(inner, "load_state_dict"):
                return inner, attr
        return None, None

    def load_finetuned(self, path: str) -> dict:
        """Load a checkpoint written by :mod:`tsfm_trl.training` into this adapter.

        Fails loudly when the keys do not match: silently keeping the base weights
        and reporting the row as "fine-tuned" is the worst possible failure mode of
        a fine-tuning library.
        """
        import torch
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
        state = obj.get("model", obj) if isinstance(obj, dict) else obj
        target, via = self._ft_target()
        if target is None or not hasattr(state, "keys"):
            raise UnavailableError(self.key,
                                   f"cannot apply fine-tuned checkpoint {path}: "
                                   f"no torch module / no state dict")
        full = dict(target.state_dict())
        missing, unexpected = target.load_state_dict(state, strict=False)
        if len(missing) == len(full):
            raise UnavailableError(
                self.key, f"fine-tuned checkpoint keys do not match this module via "
                          f"'{via}' (missing={len(missing)}, unexpected={len(unexpected)})")
        target.eval()
        self.ft_meta = obj.get("meta", {}) if isinstance(obj, dict) else {}
        self.ft_load = {"via": via, "missing": len(missing),
                        "unexpected": len(unexpected), "checkpoint": str(path)}
        return self.ft_load

    def close(self) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:                                   # noqa: BLE001
            pass


def autocast(device: str, amp: str):
    import contextlib
    if device != "cuda" or amp not in ("bf16", "fp16"):
        return contextlib.nullcontext()
    import torch
    return torch.autocast("cuda", dtype=torch.bfloat16 if amp == "bf16" else torch.float16)
