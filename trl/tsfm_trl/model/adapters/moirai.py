"""Salesforce Moirai 1.1 / Moirai-MoE / Moirai 2.0 via ``uni2ts`` (CC-BY-NC weights).

The predictor wrapper (``MoiraiForecast``/``Moirai2Forecast``) applies the model's
patch scaler, so its output is a real-space quantile grid: the same forward is used
for inference and for the fine-tune step, which keeps the optimised objective
identical to the scored one.

Note: ``uni2ts`` pins an old torch, so this family keeps its own environment
(``envs/moirai``) — see docs/01_install.md.
"""
from __future__ import annotations

import numpy as np

import torch

from ..base import LEVELS, Pred, TSFMModel, levels_from_count, to_levels


class _MoiraiBase(TSFMModel):
    caps = frozenset({"forecast", "quantiles", "multivariate", "finetune"})
    max_context = 2048
    ctx_step = 16
    loss_kind = "quantile"
    #: uni2ts names the learnable output side differently per release: 1.1/MoE put the
    #: distribution parameters in ``param_proj`` (with ``in_proj``/``out_proj`` as the
    #: multi-in/out scalers), 2.0 folds the quantile head into ``in_proj``. A heads_only
    #: fine-tune that matched nothing would either crash or silently train zero
    #: parameters, so the union is the prefix list; what actually became trainable is
    #: recorded in the result and in the checkpoint.
    head_prefixes = ("in_proj", "out_proj", "param_proj", "mask_encoding",
                     "p21", "quantile_head", "head", "output_proj",
                     "distribution_output")

    def _load(self, checkpoint):
        if "moirai-moe" in checkpoint:
            from uni2ts.model.moirai_moe import MoiraiMoEForecast as F, MoiraiMoEModule as M
        elif "2.0" in checkpoint:
            from uni2ts.model.moirai2 import Moirai2Forecast as F, Moirai2Module as M
        else:
            from uni2ts.model.moirai import MoiraiForecast as F, MoiraiModule as M
        mod = M.from_pretrained(checkpoint).to(self.device)
        if self.amp == "bf16":
            mod = mod.bfloat16()
        elif self.amp == "fp16":
            mod = mod.half()
        return F, mod

    def __init__(self, checkpoint: str = "Salesforce/moirai-1.1-R-small", **kw):
        super().__init__(**kw)
        self.checkpoint = checkpoint
        self._F, self._mod = self._load(checkpoint)
        self.patch_size = int(getattr(self._mod, "patch_size", 16))
        self.ctx_step = self.patch_size
        self.levels = self._discover_levels()

    def _discover_levels(self):
        for holder in (self._mod, getattr(self._mod, "config", None),
                       getattr(self._mod, "hparams", None)):
            for attr in ("quantiles", "quantile_levels"):
                v = getattr(holder, attr, None) if holder is not None else None
                if v is not None:
                    try:
                        out = [float(x) for x in (v.tolist() if hasattr(v, "tolist") else list(v))]
                        if len(out) >= 3:
                            return out
                    except Exception:                          # noqa: BLE001
                        pass
        return list(LEVELS)

    def _predictor(self, horizon, ctx=None):
        """uni2ts fixes the *sequence geometry* at construction: ``context_length``
        must equal the length of the tensor handed to ``forward`` (the prediction mask
        is built for that length), so the predictor is built per (horizon, context) and
        shorter contexts are left-padded with ``past_is_pad=True`` — which is exactly
        what the pad mask is for."""
        key = (int(horizon), int(ctx or self.max_context))
        cache = getattr(self, "_fcache", None)
        if cache is None:
            cache = self._fcache = {}
        if key in cache:
            return cache[key]
        # ``patch_size="auto"`` (the default) makes uni2ts loop over every candidate
        # patch size and score it — which needs a context divisible by all of them and
        # is slow. Pick the largest candidate that divides the context, and say which.
        ps = self._pick_patch(int(ctx or self.max_context), int(horizon))
        self.patch_size = ps
        self.ctx_step = ps
        kw = dict(prediction_length=int(horizon), target_dim=1, feat_dynamic_real_dim=0,
                  past_feat_dynamic_real_dim=0, context_length=int(ctx or self.max_context),
                  module=self._mod)
        try:
            f = self._F(patch_size=ps, **kw)          # Moirai 1.1 / MoE: patch is selectable
        except TypeError:                             # Moirai 2: fixed patch size
            f = self._F(**kw)
            ps = int(getattr(self._mod, "patch_size", None) or self._probe_patch())
            self.patch_size, self.ctx_step = ps, ps
        f = f.to(self.device)
        cache[key] = f
        return f

    def _probe_patch(self) -> int:
        """Moirai 2 keeps its patch size inside the module; a 1-step dry forward tells
        us how many patches the context became."""
        import torch
        n = 128
        with torch.no_grad():
            out = self._F(module=self._mod, prediction_length=16, target_dim=1,
                          feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
                          context_length=n)(torch.zeros((1, n, 1), device=self.device),
                                            torch.ones((1, n, 1), dtype=torch.bool, device=self.device),
                                            torch.zeros((1, n), dtype=torch.bool, device=self.device))
        for p in (8, 16, 32, 64, 128):
            if n % p == 0:
                return p
        return 16

    def _pick_patch(self, ctx: int, horizon: int) -> int:
        sizes = list(getattr(self._mod, "patch_sizes", None) or [16])
        ok = [p for p in sizes if ctx % p == 0]
        if not ok:
            return int(sizes[0])
        big = [p for p in ok if p <= max(16, horizon)]
        return max(big or ok)

    def _inputs(self, contexts, clen: int):
        """(past_target, past_observed_target, past_is_pad) at exactly ``clen``, with
        short contexts left-padded and flagged."""
        import torch
        cs = [np.asarray(c, dtype=float)[-clen:] for c in self.trim(contexts)]
        pt = np.zeros((len(cs), clen, 1), dtype=np.float32)
        pad = np.ones((len(cs), clen), dtype=bool)
        for i, c in enumerate(cs):
            pt[i, -len(c):, 0] = c
            pad[i, -len(c):] = False
        t = torch.tensor(pt, device=self.device)
        if self.amp == "bf16":
            t = t.bfloat16()
        obs = torch.ones_like(t, dtype=torch.bool)
        return t, obs, torch.tensor(pad, device=self.device)

    def _raw(self, f, pt, obs, pad):
        """uni2ts' forward emits (B, num_samples, H) *paths* for Moirai 1.1 and a
        quantile tensor for some Moirai 2 releases; the axis is decided from the
        shape, never assumed."""
        q = f(pt, obs, pad)
        q = q.float()
        if q.ndim == 4:
            q = q[..., 0]
        return q

    @property
    def _differentiable_head(self) -> bool:
        """Moirai 2's forward returns a quantile grid (differentiable); Moirai 1.1 and
        MoE return *samples* (not differentiable through). Decided from the tensors,
        not from a version table."""
        flag = getattr(self, "_diff_flag", None)
        if flag is not None:
            return flag
        try:
            f = self._predictor(16, 128)
            pt, obs, pad = self._inputs([np.zeros(128)], 128)
            with torch.no_grad():
                out = self._raw(f, pt, obs, pad)
            flag = bool(out.shape[1] == len(self.levels))
        except Exception:                                          # noqa: BLE001
            flag = False
        self._diff_flag = flag
        return flag

    def _q_from(self, f, pt, obs, pad):
        """Differentiable (B, H, nq). With sample output the levels are the empirical
        quantiles of the model's own paths (torch, so gradients survive)."""
        q = self._raw(f, pt, obs, pad)
        nq = len(self.levels)
        if q.shape[1] == nq:                        # already a quantile grid (B,nq,H)
            return q.transpose(1, 2)
        import torch
        lv = torch.tensor(self.levels, device=q.device, dtype=torch.float32)
        if nq == 1:
            return torch.quantile(q, float(lv[0]), dim=1).unsqueeze(-1)
        # torch.quantile puts the level axis *first*: (nq, B, H) -> (B, H, nq)
        return torch.quantile(q.float(), lv, dim=1).permute(1, 2, 0)

    def _clen(self, contexts, horizon) -> int:
        """Context length for this call: the longest trimmed context, rounded up to a
        multiple of the patch size (uni2ts reshapes context into whole patches)."""
        lens = [min(len(np.asarray(c)), self.max_context) for c in contexts]
        p = self.patch_size or 16
        return int(max(self.ctx_step, ((max(lens) + p - 1) // p) * p))

    def predict(self, contexts, horizon, *, panels=None, future=None) -> Pred:
        import torch
        clen = self._clen(contexts, horizon)
        f = self._predictor(horizon, clen)
        self._mod.eval()
        pt, obs, pad = self._inputs(contexts, clen)
        with torch.no_grad():
            q = self._q_from(f, pt, obs, pad)
        q = q.cpu().numpy()[:, :horizon, :]                        # (N,H,nq)
        nq = q.shape[-1]
        if nq == 1:
            return Pred(point=q[:, :, 0])
        lv = self.levels if len(self.levels) == nq else levels_from_count(nq)
        med = int(np.argmin(np.abs(np.asarray(lv) - 0.5)))
        return Pred(point=q[:, :, med], quantiles=to_levels(q, lv))

    def forward_quantiles(self, contexts, horizon):
        if not self._differentiable_head:
            raise NotImplementedError(f"{self.key}: sample-only head (no differentiable "
                                      f"quantile path in this uni2ts release)")
        clen = self._clen(contexts, horizon)
        f = self._predictor(horizon, clen)
        pt, obs, pad = self._inputs(contexts, clen)
        q = self._q_from(f, pt, obs, pad)
        return q[:, :horizon, :]                                   # (B,H,nq)

    def training_step(self, contexts, truth, horizon):
        import torch
        from ...training.losses import quantile_loss
        if not self._differentiable_head:
            # Moirai 1.1 / MoE: the head returns a *distribution*, and ``predict`` has to
            # sample from it — which is why reading quantiles off the samples made a
            # "fine-tune" gradient-free. The supervised route is uni2ts' own packed loss.
            return self._packed_step(contexts, truth, horizon)
        clen = self._clen(contexts, horizon)
        f = self._predictor(horizon, clen)
        pt, obs, pad = self._inputs(contexts, clen)
        q = self._q_from(f, pt, obs, pad)
        q = q[:, : len(np.asarray(truth)[0]), :]
        y = torch.tensor(np.asarray(truth, dtype=np.float32), device=self.device, dtype=q.dtype)
        lv = self.levels[: q.shape[-1]] if len(self.levels) == q.shape[-1] else \
            levels_from_count(q.shape[-1])
        return quantile_loss(q, y, lv)

    def _packed_step(self, contexts, truth, horizon):
        """Moirai 1.1 / MoE: supervise with uni2ts' own packed distribution loss.

        ``MoiraiForecast._val_loss(patch_size, target, observed_target, is_pad)`` is the
        vendor's own supervised path: it takes the whole window (context ++ future),
        splits it at the predictor's ``context_length``, runs the module, and returns the
        packed NLL over the prediction-masked patches — the objective the checkpoint was
        pretrained on, differentiable, with no sampling anywhere in the graph. The
        context is left-padded to the predictor's fixed geometry with the pad/observed
        masks set accordingly, exactly as ``predict`` does, so the tensors being optimised
        are the same ones inference sees.
        """
        import torch
        clen = self._clen(contexts, horizon)
        f = self._predictor(horizon, clen)
        ps = int(getattr(f.hparams, "patch_size", None) or self.patch_size)
        CL, PL = int(f.hparams.context_length), int(f.hparams.prediction_length)
        pt, obs, pad = self._inputs(contexts, clen)
        y = np.asarray(truth, dtype=np.float32).reshape(len(pt), -1)
        fut = np.zeros((len(pt), PL), dtype=np.float32)
        take = min(PL, y.shape[1])
        fut[:, :take] = y[:, :take]
        dt = pt.dtype
        target = torch.cat([pt, torch.tensor(fut[:, :, None], device=self.device).to(dt)], 1)
        observed = torch.cat([obs, torch.ones(len(pt), PL, 1, dtype=torch.bool,
                                              device=self.device)], 1)
        is_pad = torch.cat([pad, torch.zeros(len(pt), PL, dtype=torch.bool,
                                             device=self.device)], 1)
        fut_obs = torch.ones(len(pt), PL, 1, dtype=torch.bool, device=self.device)
        fut_pad = torch.zeros(len(pt), PL, dtype=torch.bool, device=self.device)
        try:
            from uni2ts.loss.packed import PackedNLLLoss
            packed = f._convert(ps, pt, obs, pad, future_target=target[:, CL:],
                                future_observed_target=fut_obs, future_is_pad=fut_pad)
            tgt, obs_m, sample_id, time_id, variate_id, pred_mask = packed
            distr = f.module(tgt, obs_m, sample_id, time_id, variate_id, pred_mask,
                             torch.ones_like(time_id, dtype=torch.long) * ps)
            loss = PackedNLLLoss()(pred=distr, target=tgt, prediction_mask=pred_mask,
                                   observed_mask=obs_m, sample_id=sample_id,
                                   variate_id=variate_id)
        except (ImportError, AttributeError, TypeError):
            # the fallback is the vendor's evaluation loss, whose SampleNLLLoss is a
            # Monte-Carlo estimate: unbiased, but noisy enough that a short run's curve
            # says nothing. Recorded, never silently used.
            self._loss_note = "SampleNLLLoss via _val_loss (PackedNLLLoss unavailable)"
            loss = f._val_loss(ps, target, observed, is_pad)
        return loss.mean() if loss.ndim else loss

    def trainable(self):
        # the predictor wrapper holds the module; training through it matches the
        # inference path exactly, so optimise the module itself
        return self._mod


class Moirai(_MoiraiBase):
    key, family = "moirai-1.1", "moirai"

    def __init__(self, checkpoint: str = "Salesforce/moirai-1.1-R-small", **kw):
        super().__init__(checkpoint=checkpoint, **kw)


class Moirai2(_MoiraiBase):
    key, family = "moirai-2.0", "moirai2"
    caps = frozenset({"forecast", "quantiles", "multivariate", "covariates", "finetune"})

    def __init__(self, checkpoint: str = "Salesforce/moirai-2.0-R-small", **kw):
        super().__init__(checkpoint=checkpoint, **kw)


class MoiraiMoE(_MoiraiBase):
    key, family = "moirai-moe", "moirai_moe"

    def __init__(self, checkpoint: str = "Salesforce/moirai-moe-1.0-R-base", **kw):
        super().__init__(checkpoint=checkpoint, **kw)
