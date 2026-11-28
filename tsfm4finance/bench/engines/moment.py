"""AutonLab MOMENT-1 (MIT).  The only family here with native classification and
anomaly heads, so it is the real competitor to our probe routes rather than just
another forecaster."""
from __future__ import annotations

import numpy as np
import torch

from .base import Engine, Pred


class MomentEngine(Engine):
    key, family = "moment-1-large", "moment"
    caps = {"forecast", "embed", "clf_head", "anomaly_head", "finetune"}
    max_context = 512
    max_horizon = 96

    def __init__(self, checkpoint: str = "AutonLab/MOMENT-1-large", **kw):
        super().__init__(**kw)
        from momentfm import MOMENTPipeline
        self.checkpoint = checkpoint
        self._pipe = None
        self._horizon = 0
        # set by Engine.load_finetuned; _get reads it, so it must exist before any predict call
        self._ft_load = None

    def _get(self, horizon: int):
        from momentfm import MOMENTPipeline
        if self._pipe is None or self._horizon != horizon:
            if self._ft_load is not None:
                # A fine-tuned MOMENT state dict was trained at one forecast_horizon, and the
                # pipeline is rebuilt from base weights whenever the horizon changes. Silently
                # rebuilding would then report an "adapted" cell that is really the base model.
                raise RuntimeError(
                    f"{self.key}: adapted checkpoint is fixed at horizon {self._horizon}; "
                    f"refusing to rebuild the pipeline at {horizon} (that would silently drop the "
                    "fine-tuned weights). Score this checkpoint at the horizon it was trained at.")
            p = MOMENTPipeline.from_pretrained(
                self.checkpoint,
                model_kwargs={"task_name": "forecasting", "n_channels": 1,
                              "forecast_horizon": horizon})
            p.init()
            self._pipe = p.to(self.device)
            self._horizon = horizon
        if self.amp in ("bf16", "fp16"):
            self._pipe = (self._pipe.bfloat16() if self.amp == "bf16" else self._pipe.half())
        self._pipe.eval()
        return self._pipe

    def _x(self, contexts):
        cs = self.trim(contexts)
        return torch.tensor(np.stack(cs)[:, None, :], dtype=torch.float32, device=self.device)

    def predict(self, contexts, horizon, *, panels=None) -> Pred:
        p = self._get(horizon)
        x = self._x(contexts)
        with torch.no_grad():
            out = p(x_enc=x)
        pred = getattr(out, "forecast", None)
        if pred is None:
            pred = getattr(out, "prediction", None)
        pred = np.asarray(pred.detach().float().cpu())
        pred = pred.reshape(pred.shape[0], -1)[:, :horizon]
        return Pred(point=pred, meta={"reconstruction": tuple(np.asarray(
            getattr(out, "reconstruction", np.empty(0))).shape)})

    def encode(self, contexts, horizon: int = 96) -> np.ndarray:
        p = self._get(horizon)
        with torch.no_grad():
            o = p(x_enc=self._x(contexts))
        e = getattr(o, "embeddings", None)
        if e is None:
            e = getattr(o, "embedding", None)
        return np.asarray(e.detach().float().cpu()).reshape(len(contexts), -1)

    def _ft_target(self):
        """MOMENT holds its network in a lazily built pipeline, so the generic attribute scan finds
        nothing and every adapted MOMENT cell dies with "no torch module". Build the pipeline at the
        checkpoint's own horizon, then hand back the inner module."""
        if self._pipe is None:
            self._get(min(self.max_horizon, 96))
        inner = getattr(self._pipe, "model", self._pipe)
        return (inner if hasattr(inner, "load_state_dict") else self._pipe), "_pipe"

    def classify(self, contexts, labels: np.ndarray, kind: str = "head") -> np.ndarray:
        """Native MOMENT classification head, scored **out of fold**.

        Two facts decide the shape of this method. First, the worker scores whatever probabilities
        come back, so returning in-sample predictions would be a leak dressed as a result: the head
        is fitted on K-1 folds and asked for the held-out fold, exactly like the embedding probe.
        Second, a MOMENT checkpoint carries no trained classification head - `init()` creates it at
        random - so the row is labelled `head_from_checkpoint: False` and the number is read as
        "task-adapted head", never as a pretrained capability.
        """
        from momentfm import MOMENTPipeline
        labels = np.asarray(labels)
        n_cls = int(len(np.unique(labels)))
        n_cls = max(2, n_cls)
        X = np.stack([np.asarray(c, dtype=np.float32) for c in contexts])
        n = len(X)
        k = int(min(5, max(2, n // 4)))
        idx = np.arange(n)
        rng = np.random.RandomState(0)
        rng.shuffle(idx)
        folds = np.array_split(idx, k)
        proba = np.zeros((n, n_cls), dtype=float)
        for f in range(k):
            test = folds[f]
            train = np.concatenate([folds[j] for j in range(k) if j != f])
            p = MOMENTPipeline.from_pretrained(
                self.checkpoint,
                model_kwargs={"task_name": "classification", "n_channels": 1, "n_class": n_cls})
            p.init()                       # the head is new: no trained classifier in the checkpoint
            p = p.to(self.device)
            p.train()
            opt = torch.optim.AdamW([q for q in p.parameters() if q.requires_grad], lr=1e-4)
            lossf = torch.nn.CrossEntropyLoss()
            xt = torch.tensor(X[train][:, None, :], dtype=torch.float32, device=self.device)
            yt = torch.tensor(labels[train], dtype=torch.long, device=self.device)
            for _ in range(3):
                for i in range(0, len(xt), 16):
                    opt.zero_grad()
                    out = p(x_enc=xt[i:i + 16])
                    logits = getattr(out, "logits", None)
                    if logits is None:
                        logits = out
                    lossf(logits, yt[i:i + 16]).backward()
                    opt.step()
            p.eval()
            with torch.no_grad():
                o = p(x_enc=torch.tensor(X[test][:, None, :], dtype=torch.float32,
                                         device=self.device))
                lg = getattr(o, "logits", o)
                proba[test] = torch.softmax(lg.float().cpu(), dim=1).numpy()
        self.clf_meta = {"head_from_checkpoint": False, "folds": k, "epochs": 3, "lr": 1e-4}
        return proba

    def anomaly_scores(self, contexts, horizon: int = 32) -> np.ndarray:
        p = self._get(horizon)
        x = self._x(contexts)
        with torch.no_grad():
            out = p(x_enc=x)
        rec = np.asarray(out.reconstruction.detach().float().cpu())
        inp = np.asarray(x.detach().float().cpu())
        m = np.asarray(out.mask.detach().float().cpu()) if getattr(out, "mask", None) is not None else None
        d = (rec - inp) ** 2
        if m is not None:
            d = np.where(m > 0, d, np.nan)
        return np.nanmean(d.reshape(d.shape[0], -1), axis=1)

    def trainable(self, horizon: int = 21):
        return self._get(horizon).module if hasattr(self._get(horizon), "module") else self._get(horizon)
