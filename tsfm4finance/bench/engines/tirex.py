"""NXAI TiRex (xLSTM, v1 community licence) and TiRex-2 (Apache).  Both emit a
dense quantile grid; we interpolate to the benchmark levels."""
from __future__ import annotations

import numpy as np

from .base import LEVELS, Engine, Pred, to_levels


class TiRexEngine(Engine):
    key, family = "tirex", "tirex"
    caps = {"forecast", "quantiles", "finetune"}
    max_context = 2048

    def __init__(self, checkpoint: str = "NX-AI/TiRex", **kw):
        super().__init__(**kw)
        import tirex
        self.model = tirex.load_model(checkpoint, device=self.device)
        self.max_context = int(getattr(self.model, "max_context_length", 2048))
        self.levels = self._levels()

    def _levels(self):
        for holder in (getattr(self.model, "config", None), self.model,
                       getattr(self.model, "model", None),
                       getattr(getattr(self.model, "model", None), "config", None),
                       getattr(getattr(self.model, "model", None), "checkpoint", None)):
            for attr in ("quantiles", "quantile_levels"):
                v = getattr(holder, attr, None) if holder is not None else None
                if v is not None:
                    try:
                        out = [float(x) for x in (v.tolist() if hasattr(v, "tolist") else list(v))]
                        if len(out) >= 3:
                            return out
                    except Exception:                                  # noqa: BLE001
                        pass
        return list(np.linspace(0.001, 0.999, 1000))

    def _raw(self, cs, horizon):
        """Returns (quantiles (N,h,nq), point (N,h) or None)."""
        out = self.model.forecast(cs, output_type="numpy", batch_size=self.batch)
        if isinstance(out, tuple):            # TiRex v1: (quantiles, means)
            q, mean = out
            q = np.asarray(q, float)
            if q.ndim == 2:
                return q[:, :, None], np.asarray(mean, float)
            return q, np.asarray(mean, float)
        q = np.asarray(out, float)
        if q.ndim == 2:
            return q[:, :, None], q
        return q, None

    def predict(self, contexts, horizon, *, panels=None) -> Pred:
        """TiRex emits one output patch (32 steps) per pass; longer horizons are
        produced by rolling the median back into the context.  Quantiles of the
        rolled-out steps therefore understate uncertainty growth."""
        cs = self.trim(contexts)
        q0, _ = self._raw(cs, horizon)
        step = q0.shape[1]
        qs, pts = [q0], [q0[:, :, int(np.argmin(np.abs(np.asarray(self.levels) - 0.5)))]]
        while step < horizon:
            grown = [np.concatenate([c, p[:min(step, horizon)]]) for c, p in zip(cs, pts)]
            grown = [c[-self.max_context:] for c in grown]
            q, _ = self._raw(grown, horizon)
            qs.append(q)
            pts = [np.concatenate([p, qq[:, :, int(np.argmin(np.abs(np.asarray(self.levels) - 0.5)))].squeeze(-1) if qq.ndim == 2 else qq[:, :, int(np.argmin(np.abs(np.asarray(self.levels) - 0.5)))]])
                   for p, qq in zip(pts, qs)]
            step += q.shape[1]
        full = np.concatenate(qs, axis=1)[:, :horizon, :]          # (N,H,nq)
        nq = full.shape[-1]
        lv = self.levels if len(self.levels) == nq else list(np.linspace(0.001, 0.999, nq))
        med = int(np.argmin(np.abs(np.asarray(lv) - 0.5)))
        return Pred(point=full[:, :, med], quantiles=to_levels(full, lv),
                    meta={"rollout_steps": int(np.ceil(horizon / max(step, 1)))})


def _use_native_mlstm_kernels() -> None:
    """Force mlstm_kernels' pure-PyTorch kernels.

    TiRex-2's CUDA default uses Triton kernels that do not compile against the
    Triton/CUDA pair in some environments (observed: Triton 3.9 + CUDA 13.2
    ``CompilationError`` inside the mLSTM chunk kernel).  The native kernels are
    slower but numerically equivalent, which keeps the model runnable instead of
    silently dropping it from the grid.
    """
    import tirex2.model.component.mlstm_block as MB
    orig = MB.mLSTMBackendConfig

    def native(*a, **k):
        k.update(chunkwise_kernel="chunkwise--native_autograd",
                 sequence_kernel="native_sequence__native", step_kernel="native",
                 autocast_kernel_dtype="float32")
        return orig(*a, **k)

    MB.mLSTMBackendConfig = native


class TiRex2Engine(TiRexEngine):
    key, family = "tirex-2", "tirex2"

    def __init__(self, checkpoint: str = "NX-AI/TiRex-2", kernels: str = "auto", **kw):
        Engine.__init__(self, **kw)
        import torch
        import tirex2
        self.checkpoint = checkpoint
        self.kernels = kernels
        if kernels == "native":
            _use_native_mlstm_kernels()
        self.model = tirex2.load_model(ckpt_path=checkpoint, device=self.device)
        if kernels == "auto":
            try:
                self._raw_probe()
            except Exception:                                          # noqa: BLE001
                _use_native_mlstm_kernels()
                self.model = tirex2.load_model(ckpt_path=checkpoint, device=self.device)
                self.kernels = "native"
        self.max_context = int(getattr(self.model, "max_context_length", 2048))
        self.levels = TiRexEngine._levels(self)

    def _raw_probe(self):
        import torch
        from tirex2.model.types import TimeseriesType
        cs = [np.linspace(0.0, 1.0, 128)]
        batch = [TimeseriesType(target=torch.tensor(c, dtype=torch.float32, device=self.device),
                                past_covariates=None, future_covariates=None) for c in cs]
        self.model.forecast(batch, prediction_length=8, output_type="numpy", batch_size=1)

    def predict(self, contexts, horizon, *, panels=None) -> Pred:
        import torch
        from tirex2.model.types import TimeseriesType
        cs = self.trim(contexts)
        batch = [TimeseriesType(target=torch.tensor(c, dtype=torch.float32, device=self.device),
                                past_covariates=None, future_covariates=None) for c in cs]
        out = self.model.forecast(batch, prediction_length=horizon,
                                  output_type="numpy", batch_size=self.batch)
        q = np.asarray(out, float)
        if q.ndim == 2:
            return Pred(point=q[:, :horizon], meta={"kernels": self.kernels})
        if q.shape[1] != horizon:                       # (N, nq, H) layouts vary
            q = np.transpose(q, (0, 2, 1))
        q = q[:, :horizon, :]
        nq = q.shape[-1]
        lv = self.levels if len(self.levels) == nq else list(np.linspace(0.001, 0.999, nq))
        return Pred(point=q[:, :, int(np.argmin(np.abs(np.asarray(lv) - 0.5)))],
                    quantiles=to_levels(q, lv), meta={"kernels": self.kernels})
