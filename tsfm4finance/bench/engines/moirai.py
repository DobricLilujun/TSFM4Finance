"""Salesforce Moirai 2.0 / Moirai 1.1 / Moirai-MoE via uni2ts (CC-BY-NC weights).

Moirai2's head emits nine quantiles directly (probe: output (N, 9, H)), which is
already the benchmark grid; the adapter verifies the levels before trusting it.
"""
from __future__ import annotations

import numpy as np
import torch

from .base import LEVELS, Engine, Pred, to_levels


class MoiraiEngine(Engine):
    family = "moirai"
    caps = {"forecast", "quantiles", "multivariate", "covariates", "finetune"}
    max_context = 2048
    ctx_step = 16

    def __init__(self, checkpoint: str = "Salesforce/moirai-2.0-R-small",
                 key: str | None = None, **kw):
        super().__init__(**kw)
        if "moirai-moe" in checkpoint:
            from uni2ts.model.moirai_moe import MoiraiMoEForecast as F, MoiraiMoEModule as M
        elif "2.0" in checkpoint:
            from uni2ts.model.moirai2 import Moirai2Forecast as F, Moirai2Module as M
        else:
            from uni2ts.model.moirai import MoiraiForecast as F, MoiraiModule as M
        self.checkpoint = checkpoint
        mod = M.from_pretrained(checkpoint).to(self.device)
        if self.amp in ("bf16", "fp16"):
            mod = mod.bfloat16() if self.amp == "bf16" else mod.half()
        self.module = mod
        self._h = 0
        self._F, self._mod = F, mod
        self.key = key or checkpoint.split("/")[-1]
        self.patch_size = int(getattr(mod, "patch_size", 16))
        self.ctx_step = self.patch_size
        self._levels = self._discover_levels()

    def _discover_levels(self) -> list[float]:
        for holder in (self.module, getattr(self.module, "config", None),
                       getattr(self.module, "hparams", None)):
            for attr in ("quantiles", "quantile_levels"):
                v = getattr(holder, attr, None) if holder is not None else None
                if v is not None:
                    try:
                        return [float(x) for x in (v.tolist() if hasattr(v, "tolist") else list(v))]
                    except Exception:                                  # noqa: BLE001
                        pass
        return list(LEVELS)

    def _forecast(self, contexts, horizon):
        F = self._F
        f = F(module=self._mod, prediction_length=horizon, target_dim=1,
              feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
              context_length=self.max_context)
        f = f.to(self.device)
        f.eval()
        pt = torch.tensor(np.stack(self.trim(contexts))[:, :, None], dtype=torch.float32,
                          device=self.device)
        if self.amp == "bf16":
            pt = pt.bfloat16()
        obs = torch.ones_like(pt, dtype=torch.bool)
        pad = torch.zeros(pt.shape[:2], dtype=torch.bool, device=self.device)
        with torch.no_grad():
            q = f(pt, obs, pad)                       # (N, nq, H) or (N, nq, H, 1)
        q = q.float().cpu().numpy()
        if q.ndim == 4:
            q = q[:, :, :, 0]
        return q

    def predict(self, contexts, horizon, *, panels=None) -> Pred:
        q = self._forecast(contexts, horizon)          # (N, nq, H)
        q = np.transpose(q, (0, 2, 1))[:, :horizon, :]  # (N, H, nq)
        nq = q.shape[-1]
        if nq == 1:
            return Pred(point=q[:, :, 0])
        lv = self._levels if len(self._levels) == nq else list(np.linspace(0.0, 1.0, nq + 2)[1:-1])
        med = int(np.argmin(np.abs(np.asarray(lv) - 0.5)))
        return Pred(point=q[:, :, med], quantiles=to_levels(q, lv))

    def trainable(self):
        return self.module
