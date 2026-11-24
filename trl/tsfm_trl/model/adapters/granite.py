"""IBM Granite TSFM families from ``granite-tsfm``: TinyTimeMixer (TTM r1/r2/r3),
FlowState-R1 (SSM encoder + functional-basis decoder), PatchTST-FM.

These are ordinary HuggingFace models with ``forward(past_values, future_values)``
returning a loss, which makes them the cleanest full-fine-tune targets in the
library. The released caveats are encoded here rather than discovered at run time:
r2/r3 checkpoints have a *fixed* prediction length (horizons beyond it are produced
by feeding predictions back, the vendor's own recipe), the small TTM variants ship a
mean-only head (so their rows carry no quantile columns), and FlowState publishes
its intervals under ``quantile_outputs`` with the level axis first.
"""
from __future__ import annotations

import inspect

import numpy as np

from ..base import Pred, TSFMModel, levels_from_count, to_levels


class torch_no_grad:
    """``torch.no_grad()`` without importing torch at module import time (the core
    install is torch-free on purpose)."""

    def __enter__(self):
        import torch
        self._cm = torch.no_grad()
        return self._cm.__enter__()

    def __exit__(self, *a):
        return self._cm.__exit__(*a)


def _load_keep_info(cls, checkpoint: str, kw: dict | None = None):
    """``from_pretrained`` that keeps the loading report.

    The report is what lets the adapter tell a missing head from a loaded one, which in
    turn decides whether it may print an interval at all.
    """
    try:
        model, info = cls.from_pretrained(checkpoint, output_loading_info=True, **(kw or {}))
        return model, info
    except TypeError:                                          # older granite-tsfm
        return cls.from_pretrained(checkpoint, **(kw or {})), {}


# the target kwarg name differs across transformers releases
_FUTURE_KW = ("future_values", "future_patches", "target_patches", "future_target",
              "target_values", "labels")


class _GraniteBase(TSFMModel):
    caps = frozenset({"forecast", "finetune"})
    max_context = 512
    loss_kind = "quantile"
    head_prefixes = ("head", "decoder", "linear_head", "patch_scratch", "patch_projector")

    # ------------------------------------------------------------------ setup
    def _init_from(self, cfg):
        """Read the vendor config once and pin the adapter to it."""
        # Which levels the *checkpoint* declares. If it declares none, the head emits
        # distribution parameters (student-t df/loc/scale), not a quantile grid, and
        # inventing a 9-level grid from the parameter count would print an interval that
        # the model never predicted.
        self._head_levels = [float(x) for x in (getattr(cfg, "quantiles", None) or [])]
        info = getattr(self, "loading_info", None) or {}
        miss = [str(k) for k in info.get("missing_keys", [])]
        self.head_missing = [k for k in miss
                             if any(t in k.lower() for t in ("head", "quantile", "decoder"))]
        self._fwd_params = set(inspect.signature(self.model.forward).parameters)
        self._wants_loss = "return_loss" in self._fwd_params
        pl = int(getattr(cfg, "prediction_length", 96) or -1)
        if pl > 0:
            self.pred_len = pl
        else:
            # FlowState's config says "flexible" (-1); the real chunk length is a
            # property of the checkpoint, so discover it with one dry forward rather
            # than assuming a number (a wrong guess makes the loss broadcast-fail)
            self.pred_len = self._probe_head_len()
        self.max_horizon = max(512, self.pred_len)
        self.max_context = min(self.max_context, int(getattr(cfg, "context_length", 512) or 512))
        self.ctx_step = int(getattr(cfg, "patch_length", 8) or 8)
        self.levels = list(self._head_levels)
        self.freq_name = getattr(cfg, "frequency", "h")
        if self.head_missing:
            # e.g. granite-timeseries-ttm-r3 ships quantile-head tensors that its
            # released config does not describe, so from_pretrained newly initialises
            # them; those weights are random, so the interval they would produce is
            # noise with a decimal point. Say so instead of printing it.
            self.caps = frozenset(set(self.caps) - {"quantiles"})
            self.supports["quantiles"] = (f"no: {len(self.head_missing)} head tensors were "
                                          "newly initialised at load (the released config "
                                          "does not describe them)")

    def _probe_head_len(self) -> int:
        import torch
        with torch.no_grad():
            out = self._fwd(torch.zeros((1, self.ctx_step * 4, 1), dtype=torch.float32,
                                        device=self.device))
        return int(np.asarray(out.prediction_outputs.shape).reshape(-1)[1])

    # ------------------------------------------------------------------ tensors
    def _past(self, contexts):
        import torch
        cs = self.trim(contexts)
        return torch.tensor(np.stack(cs)[:, :, None], dtype=torch.float32, device=self.device)

    def _fwd(self, past, future=None):
        import torch
        kw = {}
        if "frequency" in self._fwd_params:
            kw["frequency"] = torch.zeros((past.shape[0], 1), dtype=torch.long,
                                          device=self.device)
        if future is not None:
            # the target kwarg is not the same name across transformers releases
            # ("future_values" through 4.x, renamed again in 5.x); passing a kwarg the
            # installed forward does not declare turns a fine-tune row into a TypeError,
            # so pick the one the signature actually has
            name = next((k for k in _FUTURE_KW if k in self._fwd_params), None)
            if name is None:
                from ...errors import CapabilityError
                raise CapabilityError(
                    self.key, "finetune",
                    f"forward({sorted(self._fwd_params)}) declares no target argument in "
                    f"transformers {__import__('transformers').__version__}")
            kw[name] = future
        if self._wants_loss and future is not None:
            kw["return_loss"] = True
        return self.model(past_values=past, **kw)

    def _q_from_out(self, out, detach: bool = False):
        """Quantile head → torch (N,H,nq), or ``None`` when the checkpoint is mean-only.

        Two names across releases (``quantiles`` / ``quantile_outputs``) and, in
        FlowState, the level axis comes *before* the horizon — silently reading the
        wrong axis would turn every interval metric into nonsense, so the layout is
        normalised here and now."""
        qs = getattr(out, "quantiles", None)
        if qs is None:
            qs = getattr(out, "quantile_outputs", None)
        if qs is None:
            return None
        q = qs.detach().float() if detach else qs.float()
        if q.ndim == 4:                                   # (N, nq, H, 1)
            q = q[..., 0]
        if q.ndim == 2:
            q = q[:, None, None]
        nl = len(self.levels)
        if q.shape[1] == nl and q.shape[2] != nl:
            q = q.transpose(1, 2)                         # -> (N, H, nq)
        if not self._head_levels and q.shape[-1] != nl:
            return None            # distribution parameters, not a quantile grid
        if q.shape[-1] != nl:
            self.levels = levels_from_count(int(q.shape[-1]))
        return q

    # ------------------------------------------------------------------ inference
    def predict(self, contexts, horizon, *, panels=None, future=None) -> Pred:
        """Horizons longer than the head are produced autoregressively — the vendor's
        own recipe for TTM — with the model's predictions fed back as context."""
        import torch
        horizon = int(horizon)
        cs = self._past(contexts)
        n = cs.shape[0]
        cur, P, Q, got = cs, [], [], 0
        with torch_no_grad():
            while got < horizon:
                out = self._fwd(cur)
                mean = out.prediction_outputs.detach().float().reshape(n, -1)
                take = min(mean.shape[1], horizon - got)
                if take <= 0:
                    break
                P.append(mean[:, :take])
                q = self._q_from_out(out, detach=True)
                if q is not None:
                    Q.append(q[:, :take, :])
                got += take
                if got < horizon:
                    cur = torch.cat([cur, mean[:, :take, None]], dim=1)[:, -self.max_context:]
        point = torch.cat(P, dim=1).cpu().numpy()[:, :horizon]
        pred = Pred(point=point, meta={"pred_len": self.pred_len, "ar_chunks": len(P),
                                       "head": ("quantile" if self._head_levels
                                                else "mean/distribution parameters "
                                                     "(checkpoint declares no quantile levels)")})
        if len(Q) == len(P) and Q:
            q = torch.cat(Q, dim=1).cpu().numpy()[:, :horizon, :]
            pred.quantiles = to_levels(q, self.levels)
        return pred

    # ------------------------------------------------------------------ training
    def training_step(self, contexts, truth, horizon):
        import torch
        past = self._past(contexts)
        y = np.asarray(truth, dtype=np.float32)
        if y.ndim == 1:
            y = y[None, :]
        pad = self.pred_len - y.shape[1]
        if pad > 0:                       # heads are fixed-length: pad, don't truncate
            y = np.concatenate([y, np.repeat(y[:, -1:], pad, axis=1)], axis=1)
        elif pad < 0:                     # head is shorter than the target: supervise the head
            y = y[:, :self.pred_len]
        fut = torch.tensor(y[:, :, None], device=self.device, dtype=torch.float32)
        out = self._fwd(past, fut)
        loss = getattr(out, "loss", None)
        if loss is not None:
            return loss
        q = self._q_from_out(out)          # NOT detached: this is the training graph
        if q is not None:
            from ...training.losses import quantile_loss
            return quantile_loss(q[:, : y.shape[1], :], fut[:, :, 0], self.levels)
        from ...training.losses import huber_point
        pred = out.prediction_outputs.reshape(n := out.prediction_outputs.shape[0], -1)
        return huber_point(pred[:, : y.shape[1]], fut[:, :, 0],
                           beta=max(1e-3, float(np.nanmean(np.abs(y)))))

    def forward_prediction(self, contexts, horizon):
        """Differentiable mean head (TTM/FlowState/PatchTST all have one)."""
        out = self._fwd(self._past(contexts))
        return out.prediction_outputs.float().reshape(out.prediction_outputs.shape[0], -1)[:, :horizon]

    def forward_quantiles(self, contexts, horizon):
        """Differentiable quantile head (reward descent)."""
        q = self._q_from_out(self._fwd(self._past(contexts)))
        if q is None:
            raise NotImplementedError(f"{self.key}: checkpoint emits no quantile head")
        return q.float()[:, :horizon, :]

    def trainable(self):
        return self.model


# ---------------------------------------------------------------------------- TTM
class TTM(_GraniteBase):
    key, family = "ttm-r2", "ttm"
    caps = frozenset({"forecast", "finetune"})
    head_prefixes = ("head", "decoder")

    def __init__(self, checkpoint: str = "ibm-granite/granite-timeseries-ttm-r2",
                 variant: str | None = None, **kw):
        super().__init__(**kw)
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
        self.checkpoint = checkpoint
        self.model, self.loading_info = _load_keep_info(
            TinyTimeMixerForPrediction, checkpoint,
            {"variant": variant} if variant else {})
        self.model = self.model.to(self.device)
        self.model.eval()
        self._init_from(self.model.config)


# ---------------------------------------------------------------------- FlowState
class FlowState(_GraniteBase):
    key, family = "flowstate-r1", "flowstate"
    caps = frozenset({"forecast", "quantiles", "finetune"})
    max_context = 1536
    head_prefixes = ("decoder", "head")

    def __init__(self, checkpoint: str = "ibm-granite/granite-timeseries-flowstate-r1", **kw):
        super().__init__(**kw)
        from tsfm_public.models.flowstate import FlowStateForPrediction
        self.checkpoint = checkpoint
        self.model, self.loading_info = _load_keep_info(FlowStateForPrediction, checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()
        self._init_from(self.model.config)


# --------------------------------------------------------------------- PatchTST-FM
class PatchTSTFM(_GraniteBase):
    key, family = "patchtst-fm-r2", "patchtst"
    caps = frozenset({"forecast", "quantiles", "finetune"})
    # tsfm_public's PatchTSTFMForPrediction is one child ("backbone") whose head is
    # "backbone.out_layer"; the generic ("head","decoder") prefixes match nothing, so
    # peft="heads_only" (the reward ladder profile) failed with a misleading message
    head_prefixes = ("out_layer", "in_layer", "head", "decoder")

    def __init__(self, checkpoint: str = "ibm-granite/granite-timeseries-patchtst", **kw):
        super().__init__(**kw)
        from tsfm_public.models.patchtst_fm import PatchTSTFMForPrediction
        self.checkpoint = checkpoint
        self.model, self.loading_info = _load_keep_info(PatchTSTFMForPrediction, checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()
        self._init_from(self.model.config)
