"""Amazon Chronos family: Chronos-2 (in-context/group attention), Chronos-Bolt,
Chronos-T5.  All three emit a quantile grid; the adapter maps each model's own
grid onto the benchmark LEVELS so CRPS/WQL/VaR are comparable."""
from __future__ import annotations

import numpy as np
import torch

from .base import LEVELS, Engine, Pred, to_levels


class ChronosEngine(Engine):
    family = "chronos"
    caps = {"forecast", "quantiles", "finetune"}
    max_context = 2048
    max_horizon = 256

    def __init__(self, checkpoint: str = "amazon/chronos-2", key: str | None = None, **kw):
        super().__init__(**kw)
        import chronos
        self.checkpoint = checkpoint
        cls = (chronos.Chronos2Pipeline if "chronos-2" in checkpoint
               else chronos.ChronosBoltPipeline if "bolt" in checkpoint
               else chronos.ChronosPipeline)
        dtype = torch.bfloat16 if self.amp in ("bf16", "fp16") else torch.float32
        self.pipe = cls.from_pretrained(checkpoint, device_map=self.device, dtype=dtype)
        self.key = key or checkpoint.split("/")[-1]
        self._levels = self._discover_levels()

    def _discover_levels(self) -> list[float]:
        for attr in ("quantiles", "quantile_levels"):
            v = getattr(getattr(self.pipe, "model", None), attr, None)
            if v is None:
                v = getattr(self.pipe, attr, None)
            if v is not None:
                try:
                    return [float(x) for x in (v.tolist() if hasattr(v, "tolist") else v)]
                except Exception:                                  # noqa: BLE001
                    pass
        return list(LEVELS)

    def predict(self, contexts, horizon, *, panels=None) -> Pred:
        cs = self.trim(contexts)
        # Bolt and T5 only accept 1-D tensors (their context validator asserts on
        # numpy input); Chronos-2 accepts either, so tensors are the safe common
        # denominator across the family.
        inp = [torch.tensor(c, dtype=torch.float32) for c in cs]
        out = self.pipe.predict(inp, prediction_length=horizon)
        q = np.asarray(out.detach().cpu() if isinstance(out, torch.Tensor) else out, float)
        # Chronos emits (N, nq, H) (Bolt/T5) or (N, n_var, nq, H) (Chronos-2 with
        # variates); normalise to (N, H, nq) which is what to_levels expects.
        while q.ndim > 3:
            q = q[:, 0]
        if q.ndim == 2:
            return Pred(point=q[:, :horizon])
        if q.shape[1] != len(self._levels) and q.shape[2] == len(self._levels):
            q = q.transpose(0, 2, 1)               # (N, H, nq)
        elif q.shape[1] == len(self._levels):
            q = q.transpose(0, 2, 1)               # (N, nq, H) -> (N, H, nq)
        q = q[:, :horizon, :]
        if q.shape[2] != len(self._levels):
            return Pred(point=q[:, :, 0])
        qs = to_levels(q, self._levels)
        med = int(np.argmin(np.abs(np.asarray(self._levels) - 0.5)))
        return Pred(point=q[:, :, med], quantiles=qs)

    def trainable(self):
        return self.pipe.model
