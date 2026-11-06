"""Datadog Toto 2.0 (Apache).  Patch-based with unit-variance scaling; the head
emits nine quantiles per output patch: probe shape (9, N, n_var, patches, patch)."""
from __future__ import annotations

import numpy as np
import torch

from .base import LEVELS, Engine, Pred, to_levels


class Toto2Engine(Engine):
    key, family = "toto-2-22m", "toto2"
    caps = {"forecast", "quantiles", "multivariate", "finetune"}
    max_context = 4096
    ctx_step = 32

    def __init__(self, checkpoint: str = "Datadog/Toto-2.0-22m", key: str | None = None, **kw):
        super().__init__(**kw)
        from toto2.model import Toto2Model
        self.checkpoint = checkpoint
        self.model = Toto2Model.from_pretrained(checkpoint).to(self.device)
        self.model.eval()
        cfg = self.model.config
        self.patch = int(cfg.patch_size)
        self.ctx_step = self.patch
        self.n_out_patches = int(getattr(cfg, "num_output_patches", 16))
        self.max_horizon = self.patch * self.n_out_patches
        self.levels = [float(x) for x in (getattr(cfg, "quantiles", None)
                                          or getattr(cfg, "quantile_levels", None)
                                          or np.linspace(0.1, 0.9, 9))]
        self.key = key or checkpoint.split("/")[-1]

    def predict(self, contexts, horizon, *, panels=None) -> Pred:
        """``Toto2Model.forecast`` decodes in real space (it re-applies the causal
        scaler internally); calling ``forward`` directly leaves the head output in
        asinh-scaled units, which is why the public API is used here."""
        cs = self.trim(contexts)
        ctx = torch.tensor(np.stack(cs)[:, None, :], dtype=torch.float32, device=self.device)
        ids = torch.zeros(len(cs), 1, dtype=torch.long, device=self.device)
        batch = {"target": ctx, "target_mask": torch.ones_like(ctx, dtype=torch.bool),
                 "series_ids": ids}
        with torch.no_grad():
            q = self.model.forecast(batch, horizon=horizon, decode_block_size=0,
                                    has_missing_values=False)
        q = np.asarray(q.detach().float().cpu())      # (nq,N,n_var,H) or (nq,N,H)
        nq = q.shape[0]
        if q.ndim == 4:
            q = np.transpose(q, (1, 3, 0, 2))[:, :horizon, :, 0] if q.shape[2] == 1 \
                else np.transpose(q.reshape(nq, q.shape[1], -1, q.shape[-1]), (1, 3, 0, 2))[:, :horizon, :, 0]
        else:
            q = np.transpose(q, (1, 2, 0))
        q = q[:, :horizon, :]                                   # (N, H, nq)
        lv = self.levels if len(self.levels) == nq else list(np.linspace(0.0, 1.0, nq + 2)[1:-1])
        med = int(np.argmin(np.abs(np.asarray(lv) - 0.5)))
        return Pred(point=q[:, :, med], quantiles=to_levels(q, lv))

    def trainable(self):
        return self.model
