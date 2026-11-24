"""The Forecasting Company t0 family (t0-alpha/beta/gamma), via ``tfc-t0``.

    pip install "tfc-t0[torch]"
    model = T0Forecaster.from_pretrained("theforecastingcompany/t0-gamma")
    out   = model.predict(context, horizon=64, quantile_levels=[...])   # .quantiles, .median

The public package documents inference; there is no public training forward, so
fine-tuning is reported as ``n/a`` with that reason rather than approximated.
"""
from __future__ import annotations

import numpy as np

from ..base import to_numpy, Pred, TSFMModel, to_levels


class T0(TSFMModel):
    key, family = "t0-gamma", "t0"
    caps = frozenset({"forecast", "quantiles", "covariates", "finetune"})
    loss_kind = "quantile"
    #: t0's output side is ``head`` (per-patch quantile head) and ``decoder``;
    #: ``patch_encoder``/``transformer`` are input/trunk.
    head_prefixes = ("head", "decoder")
    #: the generic attention names (q_proj/k_proj/…) do not exist here: t0's attention is
    #: one fused ``wQKV`` plus ``wO``, and its MLP is a positional ``mlp.0``/``mlp.2``
    lora_targets = ("wQKV", "wO", "mlp")
    max_context = 4096
    max_horizon = 1024
    levels = [0.01, 0.05] + [round(0.05 * i, 2) for i in range(2, 19)] + [0.95, 0.99]

    def __init__(self, checkpoint: str = "theforecastingcompany/t0-gamma", **kw):
        super().__init__(**kw)
        from t0 import T0Forecaster
        self.checkpoint = checkpoint
        self.f = T0Forecaster.from_pretrained(checkpoint)
        try:
            self.f = self.f.to(self.device)
        except Exception:                                          # noqa: BLE001
            pass
        self.patch_len = int(getattr(self.f, "patch_size", 32))
        self.t_levels = tuple(getattr(self.f.config, "quantile_levels", ()) or ())

    def predict(self, contexts, horizon, *, panels=None, future=None) -> Pred:
        import torch
        cs = self.trim(contexts)
        x = torch.tensor(np.stack(cs), dtype=torch.float32, device=self.device)
        kw = {}
        if future is not None:
            kw["future_covariates"] = torch.tensor(np.stack(future), dtype=torch.float32,
                                                   device=self.device)
        out = self.f.predict(x, horizon=int(horizon), quantile_levels=self.levels, **kw)
        q = to_numpy(getattr(out, "quantiles", None))
        med = to_numpy(getattr(out, "median", getattr(out, "point", None)))
        if q.ndim == 2:
            q = q[:, :, None]
        q = q[:, :horizon, :]
        lv = self.levels[: q.shape[-1]]
        return Pred(point=(med[:, :horizon] if med is not None else q[:, :, q.shape[-1] // 2]),
                    quantiles=to_levels(q, lv), meta={"levels_emitted": len(lv)})

    def prepare_training(self, ctx: int, horizon: int) -> None:
        """t0 patches the window, so the training window is rounded to whole patches."""
        ps = self.patch_len
        self._ft_ctx = (max(int(ctx), 2 * ps) // ps) * ps
        self._ft_note = None
        if int(horizon) > ps:
            # one pass supervises the patch *after* the last context patch, i.e. at most
            # ``patch_size`` steps; past that the vendor's own rollout (multi-step graph)
            # is needed, so the supervised block is capped rather than faked
            self._ft_note = f"supervised block capped to one patch ({ps} of {horizon} steps)"

    def trainable(self):
        # ``T0Forecaster`` is itself the nn.Module (children: head, patch_encoder,
        # transformer, decoder) — there is no ``.model`` attribute, which is why this
        # used to report no trainable module
        return self.f

    def training_step(self, contexts, truth, horizon):
        """Teacher-free next-patch quantile loss on the vendor's own differentiable pass.

        ``_single_pass`` is ``scaler.scale_input → forward → scaler.rescale_predictions``,
        so its output is in **data space** and carries a graph — the same pass
        ``predict`` runs, minus the sampling and the level extrapolation. t0's head is
        next-patch AR: ``_predict_step`` takes ``predictions[:, context_patches - 1]``,
        i.e. the patch after the last observed one, which fixes the alignment used here.
        The future is never fed, so nothing leaks: the model sees the context and is
        scored on the steps that follow it.
        """
        import torch
        from ...training.losses import quantile_loss
        x = torch.tensor(np.asarray(self.trim(contexts), dtype=np.float32),
                         device=self.device)
        if x.shape[1] != self._ft_ctx:
            x = x[:, -self._ft_ctx:]                  # whole patches only
        from t0.data import TimeSeries
        ts = TimeSeries.from_array(targets=x, horizon=0)
        pred = self.f._single_pass(ts)                # (rows, patches, patch, levels)
        q = pred[:, -1, : min(int(horizon), self.patch_len), :].float()
        y = torch.tensor(np.asarray(truth, dtype=np.float32), device=self.device)
        y = y[:, : q.shape[1]]
        lv = list(self.t_levels)[: q.shape[-1]]
        if len(lv) != q.shape[-1]:                    # levels must match the head
            lv = [round((i + 1) / (q.shape[-1] + 1), 4) for i in range(q.shape[-1])]
        return quantile_loss(q, y[:, : q.shape[1]].float(), lv)
