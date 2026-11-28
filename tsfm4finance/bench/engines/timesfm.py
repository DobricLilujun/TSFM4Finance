"""Google TimesFM 2.5 (published case study) and TimesFM 3.0 (multivariate).

2.5 keeps using the arena adapter in ``core.models`` (that is the path the paper's
numbers came from, and it exposes decoder embeddings for the probe route).
3.0 is a different model class: variate attention + a continuous quantile head,
and its weights are under the TimesFM Non-Commercial License (fine for an academic
survey; recorded in the model table).
"""
from __future__ import annotations

import numpy as np

from .base import Engine, Pred, to_levels


class TimesFM25Engine(Engine):
    key, family = "timesfm-2.5-200m", "timesfm25"
    # no "embed": the 2.5 torch wrapper publishes no embedding API, and advertising the capability
    # sent every embedding_probe cell into an AttributeError instead of the clean capability skip
    caps = {"forecast", "quantiles", "finetune"}
    max_context = 512
    max_horizon = 128

    # The vendor 2.5 head emits nine channels (0.1 ... 0.9); channel 5 is the median.
    QUANT_CHANNELS = list(range(1, 10))

    def __init__(self, checkpoint: str | None = None, *, normalize_inputs: bool = True,
                 per_core_batch_size: int = 16, torch_compile: bool = False, **kw):
        super().__init__(**kw)
        # This engine used to import `tsfm4finance.core.models.TimesFMFinetunedModel`, which does not
        # exist in the tree - so constructing TimesFM25Engine raised ImportError for *every* cell,
        # adapted or not, and the arena had no TimesFM 2.5 row at all. Talk to the vendor package
        # directly (the same call sequence the standalone harness uses), and let the shared
        # Engine.load_finetuned contract handle adapted checkpoints.
        import timesfm
        self.checkpoint = checkpoint or timesfm.TimesFM_2p5_200M_torch.DEFAULT_REPO_ID
        self._ft_checkpoint = checkpoint if checkpoint and checkpoint != self.checkpoint else None
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            self.checkpoint, torch_compile=torch_compile)
        if self._ft_checkpoint:
            self.load_finetuned(self._ft_checkpoint)
        self.model.compile(timesfm.ForecastConfig(
            max_context=self.max_context, max_horizon=self.max_horizon,
            normalize_inputs=normalize_inputs, infer_is_positive=False,
            force_flip_invariance=True, per_core_batch_size=per_core_batch_size,
            use_continuous_quantile_head=False, fix_quantile_crossing=True))
        self._normalize_inputs = normalize_inputs
        if self._ft_checkpoint:
            self.key = f"timesfm25-ft:{self._ft_checkpoint}"

    def predict(self, contexts, horizon, *, panels=None) -> Pred:
        cs = self.trim(contexts)
        pts, qs = self.model.forecast(horizon=int(horizon),
                                      inputs=[np.asarray(c, dtype=np.float64) for c in cs])
        pts = np.asarray(pts, float)[:, :horizon]
        Q = np.asarray(qs, float)[:, :horizon, :]
        # the vendor returns [median-or-mean channel, then the 0.1..0.9 quantiles]; when only the
        # quantile block comes back, use it as-is
        q9 = Q[:, :, self.QUANT_CHANNELS] if Q.shape[-1] > 9 else Q
        return Pred(point=pts,
                    quantiles=(to_levels(q9, list(np.linspace(0.1, 0.9, q9.shape[-1])))
                               if q9 is not None and q9.ndim == 3 else None),
                    meta={"normalize_inputs": self._normalize_inputs})

    def trainable(self):
        # the differentiable stack is the inner torch model, not the vendor wrapper
        return getattr(self.model, "model", self.model)


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
