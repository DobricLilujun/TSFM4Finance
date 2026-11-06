"""AutonLab MOMENT-1 (MIT).  The only family here with native classification and
anomaly heads, so it is the real competitor to our probe routes rather than just
another forecaster."""
from __future__ import annotations

import numpy as np
import torch

from .base import Engine, Pred


class MomentEngine(Engine):
    key, family = "moment-1-large", "moment"
    caps = {"forecast", "embed", "clf_head", "anomaly_head", "finetune"}
    max_context = 512
    max_horizon = 96

    def __init__(self, checkpoint: str = "AutonLab/MOMENT-1-large", **kw):
        super().__init__(**kw)
        from momentfm import MOMENTPipeline
        self.checkpoint = checkpoint
        self._pipe = None
        self._horizon = 0

    def _get(self, horizon: int):
        from momentfm import MOMENTPipeline
        if self._pipe is None or self._horizon != horizon:
            p = MOMENTPipeline.from_pretrained(
                self.checkpoint,
                model_kwargs={"task_name": "forecasting", "n_channels": 1,
                              "forecast_horizon": horizon})
            p.init()
            self._pipe = p.to(self.device)
            self._horizon = horizon
        if self.amp in ("bf16", "fp16"):
            self._pipe = (self._pipe.bfloat16() if self.amp == "bf16" else self._pipe.half())
        self._pipe.eval()
        return self._pipe

    def _x(self, contexts):
        cs = self.trim(contexts)
        return torch.tensor(np.stack(cs)[:, None, :], dtype=torch.float32, device=self.device)

    def predict(self, contexts, horizon, *, panels=None) -> Pred:
        p = self._get(horizon)
        x = self._x(contexts)
        with torch.no_grad():
            out = p(x_enc=x)
        pred = getattr(out, "forecast", None)
        if pred is None:
            pred = getattr(out, "prediction", None)
        pred = np.asarray(pred.detach().float().cpu())
        pred = pred.reshape(pred.shape[0], -1)[:, :horizon]
        return Pred(point=pred, meta={"reconstruction": tuple(np.asarray(
            getattr(out, "reconstruction", np.empty(0))).shape)})

    def encode(self, contexts, horizon: int = 96) -> np.ndarray:
        p = self._get(horizon)
        with torch.no_grad():
            o = p(x_enc=self._x(contexts))
        e = getattr(o, "embeddings", None)
        if e is None:
            e = getattr(o, "embedding", None)
        return np.asarray(e.detach().float().cpu()).reshape(len(contexts), -1)

    def classify_fit(self, contexts, labels: np.ndarray):
        """Native classification head (fine-tuned on the task labels)."""
        from momentfm import MOMENTPipeline
        n_cls = int(len(np.unique(labels)))
        p = MOMENTPipeline.from_pretrained(
            self.checkpoint,
            model_kwargs={"task_name": "classification", "n_channels": 1,
                          "n_class": max(2, n_cls)})
        p.init()
        return p.to(self.device)

    def anomaly_scores(self, contexts, horizon: int = 32) -> np.ndarray:
        p = self._get(horizon)
        x = self._x(contexts)
        with torch.no_grad():
            out = p(x_enc=x)
        rec = np.asarray(out.reconstruction.detach().float().cpu())
        inp = np.asarray(x.detach().float().cpu())
        m = np.asarray(out.mask.detach().float().cpu()) if getattr(out, "mask", None) is not None else None
        d = (rec - inp) ** 2
        if m is not None:
            d = np.where(m > 0, d, np.nan)
        return np.nanmean(d.reshape(d.shape[0], -1), axis=1)

    def trainable(self, horizon: int = 21):
        return self._get(horizon).module if hasattr(self._get(horizon), "module") else self._get(horizon)
