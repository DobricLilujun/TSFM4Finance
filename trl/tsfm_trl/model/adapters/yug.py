"""Birla AI Labs Yug (CC-BY-NC weights): 272M zero-shot probabilistic forecaster
with covariate support and a GluonTS adapter.

    pip install yug
    p  = YugPipeline.from_pretrained("birlaailabs/yug", device_map="cuda")
    fc = p.predict(context, prediction_length=H, freq="D", seed=0)   # .median, .quantile(q)
"""
from __future__ import annotations

import numpy as np

from ..base import LEVELS, Pred, TSFMModel, to_levels


class Yug(TSFMModel):
    key, family = "yug", "yug"
    caps = frozenset({"forecast", "quantiles", "covariates"})
    loss_kind = None
    max_context = 4096
    max_horizon = 512
    ctx_step = 16
    levels = list(LEVELS)

    def __init__(self, checkpoint: str = "birlaailabs/yug", freq: str = "D", **kw):
        super().__init__(**kw)
        from yug import YugPipeline
        self.checkpoint, self.freq = checkpoint, freq
        try:
            self.p = YugPipeline.from_pretrained(checkpoint, device_map=self.device)
        except TypeError:
            self.p = YugPipeline.from_pretrained(checkpoint)
        lv = getattr(getattr(self.p, "model", None), "config", None)
        lv = getattr(lv, "quantiles", None) if lv is not None else None
        if lv:
            self.levels = [float(x) for x in lv]

    def predict(self, contexts, horizon, *, panels=None, future=None) -> Pred:
        pts, qs = [], []
        for i, c in enumerate(self.trim(contexts)):
            fc = self.p.predict(np.asarray(c, dtype=np.float32), prediction_length=int(horizon),
                                freq=self.freq, seed=self.seed + i)
            pts.append(np.asarray(fc.median, float)[:horizon])
            try:
                qs.append(np.stack([np.asarray(fc.quantile(q), float)[:horizon]
                                    for q in self.levels], axis=1))
            except Exception:                                      # noqa: BLE001
                pass
        P = np.stack(pts)[:, :horizon]
        if len(qs) == len(pts) and qs:
            Q = np.stack(qs)[:, :horizon, :]
            med = int(np.argmin(np.abs(np.asarray(self.levels) - 0.5)))
            return Pred(point=Q[:, :, med], quantiles=to_levels(Q, self.levels))
        return Pred(point=P)
