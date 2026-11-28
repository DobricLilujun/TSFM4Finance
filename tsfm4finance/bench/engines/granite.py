"""IBM Granite families from ``granite-tsfm``: TinyTimeMixer (TTM r1/r2) and
FlowState-R (state-space).  Both are HF-style modules, which makes them the
easiest full-fine-tune targets after TimesFM.

TTM caveats that must be encoded, not discovered at run time: the released r2
checkpoints have a *fixed* prediction length (96) and a fixed context (512 or
96), and the small r2 variant emits a mean only -> no quantile row in the table.
"""
from __future__ import annotations

import inspect

import numpy as np
import torch

from .base import LEVELS, Engine, Pred, to_levels


class TTMEngine(Engine):
    key, family = "ttm-r2", "ttm"
    caps = {"forecast", "finetune"}
    max_context = 512

    def __init__(self, checkpoint: str = "ibm-granite/granite-timeseries-ttm-r2",
                 variant: str | None = None, **kw):
        super().__init__(**kw)
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
        self.checkpoint = checkpoint
        self.model = TinyTimeMixerForPrediction.from_pretrained(checkpoint, variant=variant) \
            if variant else TinyTimeMixerForPrediction.from_pretrained(checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()
        cfg = self.model.config
        self.pred_len = int(getattr(cfg, "prediction_length", 96))
        self.max_horizon = self.pred_len
        self.max_context = min(self.max_context, int(getattr(cfg, "context_length", 512)))
        self.ctx_step = int(getattr(cfg, "patch_length", 8)) or 8
        self.levels = [float(x) for x in (getattr(cfg, "quantiles", None) or [])]
        self.freq_name = getattr(cfg, "frequency", "h")
        self._fwd_params = set(inspect.signature(self.model.forward).parameters)

    def _past(self, contexts):
        cs = self.trim(contexts)
        return torch.tensor(np.stack(cs)[:, :, None], dtype=torch.float32, device=self.device)

    def predict(self, contexts, horizon, *, panels=None) -> Pred:
        past = self._past(contexts)
        kw = {}
        if "frequency" in self._fwd_params:
            kw["frequency"] = torch.zeros((past.shape[0], 1), dtype=torch.long,
                                          device=self.device)
        with torch.no_grad():
            out = self.model(past_values=past, **kw)
        mean = np.asarray(out.prediction_outputs.detach().float().cpu())
        mean = mean.reshape(mean.shape[0], -1)[:, :horizon]
        qs = getattr(out, "quantiles", None)
        if qs is None or not self.levels:
            return Pred(point=mean, meta={"pred_len": self.pred_len})
        q = np.asarray(qs.detach().float().cpu())           # (N, pred_len, nq)
        if q.ndim == 4:
            q = q[:, :, :, 0]
        return Pred(point=mean, quantiles=to_levels(q[:, :horizon, :], self.levels))

    def trainable(self):
        return self.model


class FlowStateEngine(Engine):
    key, family = "flowstate-r1", "flowstate"
    caps = {"forecast", "quantiles", "finetune"}
    max_context = 1536

    def __init__(self, checkpoint: str = "ibm-granite/granite-timeseries-flowstate-r1", **kw):
        super().__init__(**kw)
        from tsfm_public.models.flowstate import FlowStateForPrediction
        self.checkpoint = checkpoint
        self.model = FlowStateForPrediction.from_pretrained(checkpoint).to(self.device)
        self.model.eval()
        cfg = self.model.config
        self.pred_len = int(getattr(cfg, "prediction_length", 96))
        self.max_horizon = max(64, self.pred_len)
        self.ctx_step = int(getattr(cfg, "patch_length", 16)) or 16
        self.levels = [float(x) for x in (getattr(cfg, "quantiles", None) or [])]
        self._fwd_params = set(inspect.signature(self.model.forward).parameters)

    def predict(self, contexts, horizon, *, panels=None) -> Pred:
        cs = self.trim(contexts)
        past = torch.tensor(np.stack(cs)[:, :, None], dtype=torch.float32, device=self.device)
        kw = {}
        if "future_values" in self._fwd_params:
            kw["future_values"] = None
        with torch.no_grad():
            out = self.model(past_values=past, **kw)
        mean = np.asarray(out.prediction_outputs.detach().float().cpu()).reshape(past.shape[0], -1)
        qs = getattr(out, "quantiles", None)
        if qs is None or not self.levels:
            return Pred(point=mean[:, :horizon])
        q = np.asarray(qs.detach().float().cpu())
        if q.ndim == 4:
            q = q[:, :, :, 0]
        return Pred(point=mean[:, :horizon], quantiles=to_levels(q[:, :horizon, :], self.levels))

    def trainable(self):
        return self.model
