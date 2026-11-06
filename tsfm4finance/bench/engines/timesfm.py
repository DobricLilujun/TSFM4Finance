"""Google TimesFM 2.5 (published case study) and TimesFM 3.0 (multivariate).

2.5 keeps using the arena adapter in ``core.models`` (that is the path the paper's
numbers came from, and it exposes decoder embeddings for the probe route).
3.0 is a different model class: variate attention + a continuous quantile head,
and its weights are under the TimesFM Non-Commercial License (fine for an academic
survey; recorded in the model table).
"""
from __future__ import annotations

import numpy as np

from .base import LEVELS, Engine, Pred, to_levels


class TimesFM25Engine(Engine):
    key, family = "timesfm-2.5-200m", "timesfm25"
    caps = {"forecast", "quantiles", "embed", "finetune"}
    max_context = 512
    max_horizon = 128

    def __init__(self, checkpoint: str | None = None, *, normalize_inputs: bool = True,
                 per_core_batch_size: int = 16, **kw):
        super().__init__(**kw)
        from tsfm4finance.core.models import TimesFMFinetunedModel, TimesFMModel
        if checkpoint:
            self.model = TimesFMFinetunedModel(checkpoint=checkpoint, max_context=self.max_context,
                                               max_horizon=self.max_horizon,
                                               per_core_batch_size=per_core_batch_size,
                                               normalize_inputs=normalize_inputs)
            self.key = f"timesfm25-ft:{checkpoint}"
        else:
            self.model = TimesFMModel(max_context=self.max_context, max_horizon=self.max_horizon,
                                      per_core_batch_size=per_core_batch_size,
                                      normalize_inputs=normalize_inputs)
        self.checkpoint = checkpoint

    def predict(self, contexts, horizon, *, panels=None) -> Pred:
        cs = self.trim(contexts)
        pts, qs = self.model.forecast_batch(cs, horizon)
        return Pred(point=np.asarray(pts, float),
                    quantiles=(np.asarray(qs, float) if qs is not None else None))

    def encode(self, contexts, max_patches: int = 8) -> np.ndarray:
        return np.asarray(self.model.encode(list(contexts), max_patches=max_patches), float)

    def trainable(self):
        return self.model.module()


class TimesFM30Engine(Engine):
    key, family = "timesfm-3.0", "timesfm30"
    caps = {"forecast", "quantiles", "multivariate", "covariates", "finetune"}
    max_context = 15360
    max_horizon = 512

    def __init__(self, checkpoint: str = "google/timesfm-3.0-pytorch", **kw):
        super().__init__(**kw)
        from timesfm import TimesFM3Forecaster
        self.checkpoint = checkpoint
        self.f = TimesFM3Forecaster.from_pretrained(checkpoint, device=self.device,
                                                   per_core_batch_size=self.batch)

    def predict(self, contexts, horizon, *, panels=None) -> Pred:
        cs = self.trim(contexts)
        outs = list(self.f.predict_batch(cs, horizon=horizon, return_quantiles=True))
        pts = np.stack([np.asarray(o.forecast, float)[-horizon:] for o in outs])
        q_list = [getattr(o, "quantiles", None) for o in outs]
        qs = None
        if all(q is not None for q in q_list):
            qs = np.stack([np.asarray(q, float)[-horizon:] for q in q_list])   # (N,H,9)
        return Pred(point=pts, quantiles=qs,
                    meta={"per_core_batch_size": self.batch})

    def trainable(self):
        from timesfm import TimesFM3Torch
        return TimesFM3Torch.from_pretrained(self.checkpoint)
