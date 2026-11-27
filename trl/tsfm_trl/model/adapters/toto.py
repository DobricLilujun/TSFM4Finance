"""Datadog Toto: Toto-Open-1.0 and Toto-2.0 (Apache).

Toto scales with a **causal** scaler, so the fine-tune target for the future block
is ``asinh((y-loc)/scale)`` from context statistics only — the same transform the
decoder inverts at inference. Anything else would optimise a different objective
than the one the model is scored on, so the training step works in the model's own
scaled space and the pinball grid is the model's own.

``toto-2`` needs python ≥ 3.12 → its own environment (``envs/toto``).
"""
from __future__ import annotations

import math

import numpy as np

from ..base import Pred, TSFMModel, levels_from_count, to_levels, to_numpy


class Toto2(TSFMModel):
    key, family = "toto-2-22m", "toto2"
    caps = frozenset({"forecast", "quantiles", "multivariate", "finetune"})
    loss_kind = "quantile"
    max_context = 4096
    ctx_step = 32
    head_prefixes = ("output_head", "head", "quantile")

    def __init__(self, checkpoint: str = "Datadog/Toto-2.0-22m", **kw):
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

    def _batch(self, contexts, extra_future=None):
        import torch
        cs = self.trim(contexts)
        ctx = torch.tensor(np.stack(cs)[:, None, :], dtype=torch.float32, device=self.device)
        ids = torch.zeros(len(cs), 1, dtype=torch.long, device=self.device)
        b = {"target": ctx, "target_mask": torch.ones_like(ctx, dtype=torch.bool),
             "series_ids": ids}
        return b

    def predict(self, contexts, horizon, *, panels=None, future=None) -> Pred:
        import torch
        b = self._batch(contexts)
        with torch.no_grad():
            q = self.model.forecast(b, horizon=int(horizon), decode_block_size=0,
                                    has_missing_values=False)
        q = np.asarray(q.detach().float().cpu())            # (nq,N,n_var,H) or (nq,N,H)
        nq = q.shape[0]
        if q.ndim == 4:
            q = np.transpose(q, (1, 3, 0, 2))[:, :horizon, :, 0] if q.shape[2] == 1 else \
                np.transpose(q.reshape(nq, q.shape[1], -1, q.shape[-1]), (1, 3, 0, 2))[:, :horizon, :, 0]
        else:
            q = np.transpose(q, (1, 2, 0))
        q = q[:, :horizon, :]
        lv = self.levels if len(self.levels) == nq else levels_from_count(nq)
        med = int(np.argmin(np.abs(np.asarray(lv) - 0.5)))
        return Pred(point=q[:, :, med], quantiles=to_levels(q, lv))

    def training_step(self, contexts, truth, horizon):
        """Pinball in the model's asinh-scaled space (see module docstring)."""
        import torch
        from ...training.losses import pinball_loss
        y = np.asarray(truth, dtype=np.float32)
        H = y.shape[1]
        pad = int(math.ceil(H / self.patch) * self.patch)
        rows, tm_rows, y_rows = [], [], []
        for c, yy in zip(np.asarray(contexts, dtype=np.float32), y):
            rows.append(np.concatenate([c, np.zeros(pad)]))
            tm_rows.append(np.r_[np.ones(len(c), dtype=bool), np.zeros(pad, dtype=bool)])
            y_rows.append(np.r_[c, yy, np.zeros(pad - H)])
        t = torch.tensor(np.stack(rows)[:, None, :], dtype=torch.float32, device=self.device)
        tm = torch.tensor(np.stack(tm_rows)[:, None, :], device=self.device)
        yfull = torch.tensor(np.stack(y_rows)[:, None, :], dtype=torch.float32, device=self.device)
        ids = torch.zeros(t.shape[0], 1, dtype=torch.long, device=self.device)
        out = self.model(target=t, target_mask=tm, cpm_mask=torch.ones_like(tm),
                         series_ids=ids, num_return_steps=None)
        q = out.quantiles.float()                              # (nq,B,1,P,patch)
        nq = q.shape[0]
        q = q.reshape(nq, t.shape[0], -1).permute(1, 2, 0)     # (B,P*patch,nq)
        with torch.no_grad():
            _scaled, loc, scale = self.model.scaler(t, tm & torch.ones_like(tm))
            loc = loc[:, 0, :] if loc.ndim == 3 else loc
            scale = scale[:, 0, :] if scale.ndim == 3 else scale
        T = q.shape[1]
        loc = loc[:, -T:]
        scale = scale[:, -T:].clamp_min(1e-6)
        tgt = torch.asinh((yfull[:, 0, -T:] - loc) / scale)
        loss = pinball_loss(q[:, -T:, :], tgt[:, -T:], self.levels[:nq])
        return loss

    def trainable(self):
        return self.model


def _find_patch(bb) -> int:
    """Patch size of the Toto 1.0 backbone (asserted against in its forward)."""
    for owner in (bb, getattr(bb, "transformer", None), getattr(bb, "config", None)):
        if owner is None:
            continue
        for name in ("patch_size", "patch", "input_patch_len"):
            v = getattr(owner, name, None)
            if isinstance(v, (int, float)) and int(v) > 1:
                return int(v)
        if isinstance(owner, dict):
            for name in ("patch_size", "patch"):
                if isinstance(owner.get(name), int) and owner[name] > 1:
                    return owner[name]
    return 64                                   # the published Toto-Open-1.0 config


class Toto1(TSFMModel):
    """Toto-Open-Base-1.0 through Datadog's own ``toto-ts`` package.

    Not through ``transformers``: the repo has no remote code and transformers has no
    Toto time-series class, so the earlier version of this adapter reported the row as
    unavailable. The vendor package *is* the implementation; the pieces are:

    * ``Toto.from_pretrained(repo, map_location=...)`` (a ``ModelHubMixin`` loader — no
      ``device_map``/``dtype`` kwargs, they would be forwarded into the model config);
    * ``TotoForecaster(model=model.model)`` — the AR decoder wants the **backbone**, not
      the wrapper;
    * a ``MaskedTimeseries`` built exactly the way their own ``evaluation/fev`` pipeline
      builds one (relative timestamps, ``time_interval_seconds=1``, ``id_mask`` ones).

    Its head is a distribution over *sampled paths*, so the intervals here are
    empirical quantiles of those paths and ``Pred.meta`` says so. ``num_samples`` must be
    divisible by ``samples_per_batch`` (their assertion).
    """

    key, family = "toto-open-1.0", "toto1"
    caps = frozenset({"forecast", "quantiles", "multivariate", "finetune"})
    loss_kind = "quantile"
    max_context = 4096
    ctx_step = 32
    head_prefixes = ("output_head", "head", "quantile")

    def __init__(self, checkpoint: str = "Datadog/Toto-Open-Base-1.0",
                 num_samples: int = 100, **kw):
        super().__init__(**kw)
        from toto.inference.forecaster import TotoForecaster
        from toto.model.toto import Toto
        self.checkpoint, self.num_samples = checkpoint, int(num_samples)
        self.model = Toto.from_pretrained(checkpoint, map_location=self.device)
        self.model.eval()
        self.bb = self.model.model                      # TotoBackbone
        self.f = TotoForecaster(model=self.bb)
        # the backbone asserts that the window length is a multiple of its patch size,
        # and the patch size is not on the backbone itself — find it wherever the
        # checkpoint put it, then align context trimming and the training window to it
        self.patch = _find_patch(self.bb)
        self.ctx_step = self.patch
        self.levels = [round(0.1 * i, 2) for i in range(1, 10)]

    def _mt(self, cs):
        """MaskedTimeseries for a batch of univariate contexts (B,V=1,T)."""
        import torch
        from toto.data.util.dataset import MaskedTimeseries
        x = torch.tensor(np.asarray(cs, dtype=np.float32)[:, None, :], device=self.device)
        b, v, t = x.shape
        return MaskedTimeseries(
            series=x,
            padding_mask=torch.ones_like(x, dtype=torch.bool),
            id_mask=torch.ones_like(x, dtype=torch.long),
            timestamp_seconds=torch.arange(t, device=self.device, dtype=torch.long)
            .view(1, 1, t).expand(b, v, t),
            time_interval_seconds=torch.ones(b, v, dtype=torch.long, device=self.device),
            num_exogenous_variables=0,
        )

    def predict(self, contexts, horizon, *, panels=None, future=None) -> Pred:
        import torch
        ns = max(10, self.num_samples)
        ns -= ns % 10                                   # divisible by samples_per_batch
        with torch.no_grad():
            out = self.f.forecast(inputs=self._mt(self.trim(contexts)),
                                  prediction_length=int(horizon), num_samples=ns,
                                  samples_per_batch=10, use_kv_cache=True)
        mean = to_numpy(out.mean)[:, 0, :horizon]                       # (B,H)
        samples = to_numpy(out.samples)[:, 0, :horizon, :]              # (B,H,S)
        q = np.quantile(samples, self.levels, axis=2).transpose(1, 2, 0)
        return Pred(point=mean, quantiles=q,
                    meta={"how": f"empirical from {samples.shape[-1]} sampled paths "
                                 f"(Toto 1.0 distribution head)"})

    def training_step(self, contexts, truth, horizon):
        """The vendor's pretraining objective: ``CombinedLoss`` (NLL + general robust
        loss) on the causally-scaled window, teacher-forced over context++future. Using
        their loss keeps the fine-tune target the same objective the checkpoint was
        trained on, including the scaler's own loc/scale."""
        import torch
        from toto.model.losses import CombinedLoss
        cs = self.trim(contexts)
        y = np.asarray(truth, dtype=np.float32).reshape(len(cs), -1)
        window = np.concatenate([cs, y], axis=1)
        # teacher-forced window, right-padded to a patch multiple; the pad is masked out
        # of the loss rather than fed to the model as data it should explain
        pad = (-window.shape[1]) % self.patch
        if pad:
            window = np.concatenate([window, np.zeros((window.shape[0], pad),
                                                      dtype=np.float32)], axis=1)
        x = torch.tensor(window[:, None, :], dtype=torch.float32, device=self.device)
        pm = torch.ones_like(x, dtype=torch.bool)
        if pad:
            pm[:, 0, -pad:] = False
        out = self.bb(inputs=x, input_padding_mask=pm,
                      id_mask=torch.ones_like(x, dtype=torch.long))
        eps = torch.finfo(x.dtype).eps
        tgt = (x - out.loc) / (out.scale + eps)
        loss = CombinedLoss()(out.distribution, tgt) * pm
        return loss.sum() / (pm.sum() + eps)

    def trainable(self):
        return self.model


def _find_patch(bb) -> int:
    """Patch size of the Toto 1.0 backbone (asserted against in its forward)."""
    for owner in (bb, getattr(bb, "transformer", None), getattr(bb, "config", None)):
        if owner is None:
            continue
        for name in ("patch_size", "patch", "input_patch_len"):
            v = getattr(owner, name, None)
            if isinstance(v, (int, float)) and int(v) > 1:
                return int(v)
        if isinstance(owner, dict):
            for name in ("patch_size", "patch"):
                if isinstance(owner.get(name), int) and owner[name] > 1:
                    return owner[name]
    return 64                                   # the published Toto-Open-1.0 config
