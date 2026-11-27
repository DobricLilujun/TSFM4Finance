"""Mistral AI Timestral (modified-MIT weights, code on GitHub).

    pip install git+https://github.com/mistralai/timestral.git
    f   = TimestralForecaster.from_pretrained("mistralai/Timestral-28M")
    fc  = f.forecast(context, prediction_length=96)      # .median, .quantile(q)

Patch-based (32/patch) with time + variate attention, 21 quantile knots, up to
32,768 context steps and 1,024 horizon steps per forward. Fine-tuning: the public
release ships the inference forecaster only, so ``finetune`` is reported ``n/a``
until a differentiable forward is exposed — see docs/03_fine_tune.md.
"""
from __future__ import annotations

import numpy as np

from ..base import LEVELS, Pred, TSFMModel, to_levels, to_numpy

KNOTS = [0.01, 0.05] + [round(0.05 * i, 2) for i in range(2, 19)] + [0.95, 0.99]


class Timestral(TSFMModel):
    key, family = "timestral", "timestral"
    caps = frozenset({"forecast", "quantiles", "multivariate", "covariates"})
    loss_kind = None
    max_context = 4096
    max_horizon = 1024
    ctx_step = 32
    levels = list(KNOTS)

    def __init__(self, checkpoint: str = "mistralai/Timestral-28M", **kw):
        super().__init__(**kw)
        from timestral import TimestralForecaster
        self.checkpoint = checkpoint
        self.f = TimestralForecaster.from_pretrained(checkpoint, device=self.device) \
            if _accepts_device(TimestralForecaster.from_pretrained) else \
            TimestralForecaster.from_pretrained(checkpoint)
        try:
            self.levels = [float(x) for x in (self.f.config["quantile_knots"])] \
                if hasattr(self.f, "config") else list(KNOTS)
        except Exception:                                          # noqa: BLE001
            self.levels = list(KNOTS)

    def predict(self, contexts, horizon, *, panels=None, future=None) -> Pred:
        pts, qs = [], []
        for c in self.trim(contexts):
            fc = self.f.forecast(np.asarray(c, dtype=np.float64), prediction_length=int(horizon))
            pts.append(to_numpy(fc.median).reshape(-1)[:horizon])
            try:
                grid = np.stack([to_numpy(fc.quantile(q)).reshape(-1)[:horizon]
                                for q in self.levels], axis=1)
                qs.append(grid)
            except Exception:                                      # noqa: BLE001
                pass
        P = np.stack(pts)[:, :horizon]
        if len(qs) == len(pts) and qs:
            Q = np.stack(qs)[:, :horizon, :]
            med = int(np.argmin(np.abs(np.asarray(self.levels) - 0.5)))
            return Pred(point=Q[:, :, med], quantiles=to_levels(Q, self.levels), levels=LEVELS,
                        meta={"levels_emitted": len(self.levels)})
        return Pred(point=P)

    def trainable(self):
        return getattr(self.f, "model", None)

    def training_step(self, contexts, truth, horizon):
        from ...errors import CapabilityError
        raise CapabilityError(self.key, "finetune",
                              "the released timestral package exposes the inference forecaster "
                              "only; wire the vendor training forward (mistralai/timestral) "
                              "before claiming a Timestral fine-tune")


def _accepts_device(fn) -> bool:
    import inspect
    try:
        return "device" in inspect.signature(fn).parameters
    except Exception:                                              # noqa: BLE001
        return False
