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


def head_weights_in_checkpoint(ckpt_heads: dict, live_heads: dict) -> bool:
    """Were the head tensors this task asks for actually present in the checkpoint?

    The released MOMENT-1 checkpoints carry exactly one head: `head.linear` of shape
    (8, d_model) — the **reconstruction** head. A forecasting or classification pipeline
    asks for a different tensor, so its head is randomly initialised, and `init()` (which
    the model card tells you to call) is the call that replaces it. transformers reports
    "0 missing keys" for every task because the new head has different *shapes*, not
    because it was loaded. That is the difference between "the model ran" and "the model
    was trained to do this", and an effectiveness table must not blur it.

    Pure function, no momentfm import, so the claim is testable in the core environment.
    """
    if not live_heads:
        return False
    return all(k in ckpt_heads and list(ckpt_heads[k]) == list(live_heads[k])
               for k in live_heads)


def _checkpoint_heads(checkpoint):
    """Head tensor shapes inside a checkpoint, without loading the backbone.

    Returns None when the checkpoint cannot be inspected (hub id with no local cache,
    unusual layout): unknown is recorded as unknown, never guessed as True or False.
    """
    from pathlib import Path
    try:
        d = Path(str(checkpoint)).expanduser()
        files = []
        if d.is_dir():
            files = [d / "model.safetensors", d / "pytorch_model.bin"]
        elif d.is_file():
            files = [d]
        else:
            from huggingface_hub import hf_hub_download
            for name in ("model.safetensors", "pytorch_model.bin"):
                try:
                    files.append(Path(hf_hub_download(str(checkpoint), name)))
                    break
                except Exception:                            # noqa: BLE001
                    continue
        for f in files:
            if not f.exists():
                continue
            if f.suffix == ".safetensors":
                from safetensors import safe_open
                with safe_open(str(f), framework="pt") as h:
                    return {k: list(h.get_slice(k).get_shape())
                            for k in h.keys() if k.startswith("head")}
            import torch
            sd = torch.load(str(f), map_location="cpu", mmap=True)
            return {k: list(v.shape) for k, v in sd.items() if k.startswith("head")}
    except Exception:                                        # noqa: BLE001
        return None
    return None


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
        # NOTE: `self._pipe` / `self._key` used to live here as a single-pipeline cache
        # sentinel. They are gone because a method added later was called `_key`, and the
        # dead attribute shadowed it into `None` — every zero-shot MOMENT row then died as
        # `TypeError: 'NoneType' object is not callable` in the sweep, not at import. A
        # name that is both an attribute and a method is a latent break: keep them apart.
        self._want_ctx, self._want_horizon = 0, 0

    # ------------------------------------------------------------------ plumbing
    def prepare_training(self, ctx: int, horizon: int) -> None:
        """MOMENT's module graph (position/mask buffers, head widths) is fixed at
        ``init()`` by ``seq_len``/``pred_len``, so the trainers tell the adapter what
        shape they are about to train before asking for the module."""
        self._want_ctx, self._want_horizon = int(ctx), int(horizon)

    def _pipe_key(self, task: str, horizon: int, n_class: int, contexts=None):
        """Pipeline cache key. Built in one place because the *prediction* has to name the
        same pipeline the *training* used, or the trained/untrained label is a guess."""
        return (task, int(horizon), self._ctx_for(horizon, contexts), int(n_class))

    def _get(self, horizon: int, task: str = "forecasting", n_class: int = 2, fit: bool = False,
             contexts=None):
        """Pipelines are cached per (task, horizon, seq_len) — rebuilding a 1B model per
        call would dominate any fine-tune."""
        from momentfm import MOMENTPipeline
        ctx = self._ctx_for(horizon, contexts)
        key = self._pipe_key(task, horizon, n_class, contexts)
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
            # record whether the task head is pretrained, per pipeline (see
            # head_weights_in_checkpoint): the row then says what it actually measured
            live = {k: list(v.shape) for k, v in p.state_dict().items()
                    if k.startswith("head")}
            ck = _checkpoint_heads(self.checkpoint)
            self._head_state = getattr(self, "_head_state", {})
            self._head_state[key] = None if ck is None else head_weights_in_checkpoint(ck, live)
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
        # a green cell in the compatibility matrix only means the call returned. These
        # flags let an effectiveness table say whether the head it scored was ever
        # trained for this task (MOMENT's checkpoint ships only a reconstruction head).
        key = self._pipe_key("forecasting", horizon, 2, contexts)
        from_ckpt = getattr(self, "_head_state", {}).get(key)
        meta = {"heads": list(getattr(out, "head_names", []) or [])}
        if from_ckpt is not None:
            meta["head_from_checkpoint"] = bool(from_ckpt)
            meta["head_trained_here"] = bool(
                getattr(self, "_trained_keys", set()) & {key})
        return Pred(point=pred[:, :horizon], meta=meta)

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
        # training in this process is a fact about this pipeline, so a prediction that
        # follows it is no longer an untrained-head row
        if not hasattr(self, "_trained_keys"):
            self._trained_keys = set()
        self._trained_keys.add(self._pipe_key("forecasting", y.shape[1], 2, contexts))
        out = p(x_enc=self._x(contexts))
        pred = getattr(out, "forecast", None)
        if pred is None:
            pred = getattr(out, "prediction", None)
        pred = pred.reshape(pred.shape[0], -1)[:, : y.shape[1]].float()
        scale = max(1e-6, float(np.nanmean(np.abs(y))))
        return huber_point(pred, torch.tensor(y, device=pred.device, dtype=pred.dtype), beta=scale)
