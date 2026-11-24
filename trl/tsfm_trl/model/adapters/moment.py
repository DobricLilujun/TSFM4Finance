"""AutonLab MOMENT-1 (MIT).

The only family in the library with **native classification and anomaly heads**, so
it is the real competitor to the probe/head routes rather than just another
forecaster — that comparison is the point of having three classification paths.

``momentfm`` pins numpy/huggingface-hub tightly, so this family keeps its own
environment (``envs/moment``).
"""
from __future__ import annotations

import numpy as np

from ...errors import UnavailableError
from ..base import Pred, TSFMModel


class Moment(TSFMModel):
    key, family = "moment-1-large", "moment"
    caps = frozenset({"forecast", "embed", "clf_head", "anomaly_head", "finetune"})
    loss_kind = "quantile"                 # point head -> L1/pinball on its output
    max_context = 512
    max_horizon = 96
    ctx_step = 8
    head_prefixes = ("heads", "head", "forecast_head", "projection")
    supports = {"trunk_grad": True, "native_clf": True, "native_anomaly": True}

    def __init__(self, checkpoint: str = "AutonLab/MOMENT-1-large", **kw):
        super().__init__(**kw)
        self.checkpoint = checkpoint
        self._pipe = None
        self._key = None
        self._want_ctx, self._want_horizon = 0, 0

    # ------------------------------------------------------------------ plumbing
    def prepare_training(self, ctx: int, horizon: int) -> None:
        """MOMENT's module graph (position/mask buffers, head widths) is fixed at
        ``init()`` by ``seq_len``/``pred_len``, so the trainers tell the adapter what
        shape they are about to train before asking for the module."""
        self._want_ctx, self._want_horizon = int(ctx), int(horizon)

    def _get(self, horizon: int, task: str = "forecasting", n_class: int = 2, fit: bool = False,
             contexts=None):
        """Pipelines are cached per (task, horizon, seq_len) — rebuilding a 1B model per
        call would dominate any fine-tune."""
        from momentfm import MOMENTPipeline
        ctx = self._ctx_for(horizon, contexts)
        key = (task, int(horizon), ctx, int(n_class))
        cache = getattr(self, "_pipes", None)
        if cache is None:
            cache = self._pipes = {}
        if key not in cache:
            kw = {"task_name": task, "n_channels": 1}
            if task in ("forecasting",):
                kw.update({"seq_len": ctx, "pred_len": int(horizon),
                           "forecast_horizon": int(horizon)})
            elif task in ("reconstruction", "imputation"):
                kw["seq_len"] = ctx
            else:                                        # classification / named tasks
                # MOMENT's own spelling is num_class; "n_class must be specified" and
                # "num_class must be specified" are the two errors its versions raise
                kw["num_class"] = int(n_class)
            try:
                p = MOMENTPipeline.from_pretrained(self.checkpoint, model_kwargs=kw)
            except Exception:                                        # noqa: BLE001
                if "num_class" in kw:
                    kw2 = {**kw, "n_class": kw.pop("num_class")}
                    p = MOMENTPipeline.from_pretrained(self.checkpoint, model_kwargs=kw2)
                else:
                    raise
            p.init()
            # MOMENT's backbone turns gradient checkpointing on by default, and its
            # recomputation is not deterministic (the token-embedding branch draws a
            # different number of tensors the second time), so torch.utils.checkpoint
            # aborts with "different number of tensors was saved". Turn it off: these
            # fine-tunes are smoke-sized and memory is not the constraint.
            for holder in (p, getattr(p, "model", None)):
                for meth in ("gradient_checkpointing_disable", "disable_gradient_checkpointing"):
                    f = getattr(holder, meth, None)
                    if callable(f):
                        f()
            cache[key] = p.to(self.device)
        p = cache[key]
        if self.amp == "bf16":
            p = p.bfloat16()
        elif self.amp == "fp16":
            p = p.half()
        p.train() if fit else p.eval()
        return p

    def _ctx_for(self, horizon: int, contexts=None) -> int:
        """MOMENT's ``init()`` fixes seq_len, so the pipeline is built for the context
        length actually in front of us (rounded to a multiple of the 8-step patch).
        Getting this from the data — not from a config guess — is what makes the same
        adapter work at ctx 128 and ctx 512."""
        if contexts is not None:
            lens = [len(np.asarray(c)) for c in contexts]
            c = min(max(lens), self.max_context) if lens else self.max_context
        else:
            c = self._want_ctx or min(self.max_context, 512)
        return int(max(64, (c // 8) * 8))

    def set_context(self, ctx: int):
        self._want_ctx = int(ctx)
        return self

    def _x(self, contexts):
        import torch
        cs = self.trim(contexts)
        return torch.tensor(np.stack(cs)[:, None, :], dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------ forecast
    def predict(self, contexts, horizon, *, panels=None, future=None) -> Pred:
        import torch
        p = self._get(horizon, contexts=contexts)
        with torch.no_grad():
            out = p(x_enc=self._x(contexts))
        pred = getattr(out, "forecast", None)
        if pred is None:
            pred = getattr(out, "prediction", None)
        pred = np.asarray(pred.detach().float().cpu()).reshape(len(list(contexts)), -1)
        return Pred(point=pred[:, :horizon],
                    meta={"heads": list(getattr(out, "head_names", []) or [])})

    def encode(self, contexts, horizon: int = 96) -> np.ndarray:
        """Frozen representation: MOMENT's own ``embed`` head (the paper's masked
        representation), not a by-product of the forecast."""
        import torch
        p = self._get(horizon, contexts=contexts)
        with torch.no_grad():
            o = p.embed(x_enc=self._x(contexts))
        e = getattr(o, "embeddings", None)
        if e is None:
            raise UnavailableError(self.key, "this momentfm build returns no embeddings "
                                             "from embed()")
        return np.asarray(e.detach().float().cpu()).reshape(len(list(contexts)), -1)

    def anomaly_scores(self, contexts):
        """Model-native anomaly: MOMENT is pretrained with masked reconstruction, so the
        reconstruction residual of the observed window *is* its anomaly signal (the task
        the paper evaluates), obtained from the vendor's reconstruction task head."""
        import torch
        p = self._get(16, task="reconstruction", contexts=contexts)
        x = self._x(contexts)
        with torch.no_grad():
            o = p(x_enc=x)
        rec = getattr(o, "reconstruction", None)
        if rec is None:
            raise UnavailableError(self.key, "this momentfm build returns no "
                                             "reconstruction output")
        r = rec.detach().float()
        if r.shape[-1] != x.shape[-1]:                     # padded to seq_len
            r = r[..., -x.shape[-1]:]
        return torch.nanmean((r - x).abs(), dim=(-1, -2)).cpu().numpy()

    def classifier_features(self, contexts):
        """Differentiable embedding block (the trunk stays in the graph), which is
        what lets ``unfreeze_trunk=True`` train MOMENT end-to-end."""
        h = min(self.max_horizon, 64)
        p = self._get(h, fit=True, contexts=contexts)
        o = p.embed(x_enc=self._x(contexts))
        e = getattr(o, "embeddings", None)
        if e is None:
            e = getattr(o, "prediction", None)
        if e is None:
            raise UnavailableError(self.key, "no differentiable embedding head in this "
                                             "momentfm build")
        return e.reshape(e.shape[0], -1)

    # ------------------------------------------------------------------ native heads
    def classify_fit(self, contexts, labels):
        """MOMENT's own classification head (returned untrained-but-loaded; the
        ClassificationTrainer fits it with the same loop as every other model)."""
        n_cls = int(len(np.unique(labels)))
        return self._get(self._ctx_for(16, contexts), task="classification",
                          n_class=max(2, n_cls), fit=True, contexts=contexts)

    def forward_prediction(self, contexts, horizon):
        """Differentiable forecast tensor: reward descent on a mean-only head."""
        p = self._get(int(horizon), fit=True, contexts=contexts)
        out = p(x_enc=self._x(contexts))
        f = getattr(out, "forecast", None)
        if f is None:
            raise NotImplementedError(f"{self.key}: forecasting head returned no forecast")
        return f.float().reshape(f.shape[0], -1)[:, :int(horizon)]

    # ------------------------------------------------------------------ training
    def trainable(self):
        h = self._want_horizon or min(self.max_horizon, 96)
        return self._get(h, fit=True)

    def training_step(self, contexts, truth, horizon):
        import torch
        from ...training.losses import huber_point
        y = np.asarray(truth, dtype=np.float32)
        p = self._get(y.shape[1], fit=True, contexts=contexts)
        out = p(x_enc=self._x(contexts))
        pred = getattr(out, "forecast", None)
        if pred is None:
            pred = getattr(out, "prediction", None)
        pred = pred.reshape(pred.shape[0], -1)[:, : y.shape[1]].float()
        scale = max(1e-6, float(np.nanmean(np.abs(y))))
        return huber_point(pred, torch.tensor(y, device=pred.device, dtype=pred.dtype), beta=scale)
