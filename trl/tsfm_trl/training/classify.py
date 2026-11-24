"""``ClassificationTrainer`` — classification fine-tuning of a TSFM, three ways.

``probe``    frozen model → shallow classifier (logistic / GBM), out-of-fold.
             Works for *every* model that can forecast, and it is the baseline a
             head must beat.
``head``     trainable MLP head on the model's representation (cross-entropy with
             class weights). If the adapter supports gradients through its trunk
             (``model.supports["trunk_grad"]``), ``unfreeze_trunk=True`` trains the
             backbone too — and the checkpoint says which happened.
``native``   the vendor's own head (today: MOMENT's ``clf_head``), same loop.

All three report PR-AUC / ROC-AUC / macro-F1 on the held-out split, plus the
majority-class base rate, because the derived finance labels are imbalanced and a
classifier that predicts the majority class must not look like a result.
"""
from __future__ import annotations

import time

import numpy as np

from ..errors import CapabilityError
from ..eval.metrics import classification_report
from .config import StepLog, TrainResult, TrainingConfig


class ClassificationTrainer:
    def __init__(self, model, cfg: TrainingConfig | None = None, *, n_classes: int | None = None,
                 route: str = "head", classifier: str = "logistic",
                 unfreeze_trunk: bool = False, **cfg_kw):
        self.model = model
        self.cfg = cfg or TrainingConfig(**cfg_kw)
        self.route = route
        self.classifier = classifier
        self.unfreeze_trunk = unfreeze_trunk
        self.n_classes = n_classes

    # ------------------------------------------------------------------ probe
    def fit_probe(self, cds) -> TrainResult:
        from ..probing.features import folds, make_classifier, model_matrix, stat_matrix
        t0 = time.time()
        X, y, tr, te = cds.parts()
        use_model = self.classifier != "stat"
        F = model_matrix(self.model, X) if use_model else stat_matrix(X)
        kinds = ("logistic", "gbm") if use_model else ("stat-logistic", "stat-gbm")
        best = None
        rows = []
        for kind in kinds:
            preds = np.full(len(y), -1, dtype=int)
            S = None                                        # out-of-fold scores, if any
            for tri, tei in folds(cds.split, mode="expanding"):
                if len(np.unique(y[tri])) < 2:
                    continue
                clf = make_classifier("logistic" if kind.startswith("stat") else kind)
                clf.fit(F[tri], y[tri])
                preds[tei] = clf.predict(F[tei])
                if hasattr(clf, "predict_proba"):
                    if S is None:
                        # Pipeline-wrapped classifiers expose classes_, not n_classes_
                        S = np.full((len(y), len(np.asarray(clf.classes_))), np.nan)
                    S[tei] = clf.predict_proba(F[tei])
            keep = preds >= 0
            if not keep.any():
                continue
            rep = classification_report(y[keep], S[keep] if S is not None else preds[keep],
                                        n_classes=cds.n_classes)
            rep["features"] = "model" if use_model else "statistical"
            rep["classifier"] = kind
            rows.append(rep)
            if best is None or rep["macro_f1"] > best["macro_f1"]:
                best = rep
        if best is None:
            return TrainResult(model=self.model.key, status="failed",
                               reason="probe: no fold had >= 2 classes")
        return TrainResult(model=self.model.key, status="ok", steps=0,
                           seconds=time.time() - t0, n_windows=len(X),
                           extra={"route": "probe", "metrics": best, "all": rows})

    # ------------------------------------------------------------------ head
    def fit_head(self, cds) -> TrainResult:
        import torch
        cfg = self.cfg
        t0 = time.time()
        X, y, tr, te = cds.parts()
        n_cls = self.n_classes or cds.n_classes

        trunk_grad = bool(self.unfreeze_trunk) and bool(self.model.supports.get("trunk_grad"))
        if self.unfreeze_trunk and not trunk_grad:
            return TrainResult(model=self.model.key, status="skipped",
                               reason=f"{self.model.key}: unfreeze_trunk requested but this "
                                      f"adapter exposes no differentiable trunk "
                                      f"(supports={{'trunk_grad': False}})")

        Ftr = self._features(X, tr, torch, grad=trunk_grad)
        Fte = self._features(X, te, torch, grad=False)
        if Ftr is None or Fte is None:
            return TrainResult(model=self.model.key, status="failed",
                               reason="feature extraction produced nothing")
        dev = Ftr.device
        ytr = torch.tensor(y[tr], dtype=torch.long, device=dev)
        yte = torch.tensor(y[te], dtype=torch.long, device=dev)

        dim = int(Ftr.shape[1])
        head = torch.nn.Sequential(torch.nn.LayerNorm(dim),
                                   torch.nn.Linear(dim, 128), torch.nn.GELU(),
                                   torch.nn.Dropout(0.1),
                                   torch.nn.Linear(128, n_cls)).to(dev)
        params = list(head.parameters())
        trunk_mod = None
        if trunk_grad:
            trunk_mod = self.model.trainable()
            params += [p for p in trunk_mod.parameters() if p.requires_grad]
        counts = sum(p.numel() for p in params)

        w = torch.ones(n_cls, device=ytr.device)
        cnt = np.bincount(y[tr], minlength=n_cls).astype(float)
        w = torch.tensor((cnt.sum() / np.maximum(cnt, 1.0)) / max(1, n_cls),
                         device=ytr.device, dtype=torch.float32)
        lossf = torch.nn.CrossEntropyLoss(weight=w)

        steps = cfg.eff_steps(max(8, int(tr.sum())))
        # a from-scratch head needs a from-scratch learning rate: the budget's lr is a
        # fine-tuning lr (1e-5), and at that step size a fresh MLP never leaves its
        # initial bias — which reads as accuracy *below* chance, not as underfitting
        head_lr = float(cfg.lr) if cfg.lr else 1e-3
        groups = [{"params": list(head.parameters()), "lr": head_lr}]
        if trunk_mod is not None:
            # the backbone keeps the (much smaller) fine-tuning rate it was tuned at
            groups.append({"params": [p for p in trunk_mod.parameters() if p.requires_grad],
                           "lr": cfg.eff_lr()})
        opt = torch.optim.AdamW(groups, lr=head_lr, weight_decay=cfg.weight_decay)
        rng = np.random.RandomState(cfg.seed)
        log = StepLog(cfg.log_every)
        if trunk_mod is not None:
            trunk_mod.train()
        head.train()
        n = Ftr.shape[0]
        for step in range(1, steps + 1):
            idx = rng.choice(n, size=min(cfg.bs, n), replace=False)
            # with a trainable trunk the features must be recomputed for the batch,
            # otherwise the trunk would receive no gradient
            xb = self._features(X, tr, torch, grad=True, subset=idx) if trunk_grad else Ftr[idx]
            yb = ytr[idx] if not trunk_grad else ytr[torch.as_tensor(idx, device=dev)]
            logits = head(xb)
            loss = lossf(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            log.step(step, float(loss.item()))
        head.eval()
        with torch.no_grad():
            # scores, not argmax: PR-AUC is the metric that separates skill from a
            # majority-class guess, and it cannot be recovered from hard labels
            ste = torch.softmax(head(Fte), -1).cpu().numpy()
            str_ = torch.softmax(head(Ftr), -1).cpu().numpy()
            rep = classification_report(yte.cpu().numpy(), ste, n_classes=n_cls)
            rep_tr = classification_report(ytr.cpu().numpy(), str_, n_classes=n_cls)
        ck = ""
        if not cfg.smoke:
            from .checkpoints import save_ckpt
            meta = {"model": self.model.key, "route": "head", "dataset": cds.name,
                    "n_classes": n_cls, "feature_dim": dim, "trunk_grad": trunk_grad,
                    "trainable_parameters": counts, "budget": cfg.budget,
                    "split": cfg.split, "loss_first": log.first, "loss_last": log.last,
                    "metrics_test": rep, "metrics_train": rep_tr, "smoke": cfg.smoke,
                    "seed": cfg.seed}
            ck = save_ckpt(cfg.ckpt_path(f"{self.model.key}__clf"), head, meta,
                           cfg.to_dict(), log.steps)
        return TrainResult(model=self.model.key, status="smoke" if cfg.smoke else "ok",
                           steps=steps, first_loss=log.first, final_loss=log.last,
                           checkpoint=ck, trainable_parameters=counts,
                           n_windows=int(tr.sum()), seconds=time.time() - t0,
                           extra={"route": "head", "metrics": rep, "train_metrics": rep_tr,
                                  "trunk_grad": trunk_grad})

    def _features(self, X, mask, torch, *, grad: bool, subset=None):
        idx = np.flatnonzero(mask)
        if subset is not None:
            idx = idx[subset]
        if len(idx) == 0:
            return None
        # elementwise: X is a container of variable-length windows, and a plain python
        # list does not accept the numpy index the object array does
        windows = [X[int(i)] for i in idx]
        dev = self.cfg.device if self.cfg.device not in ("", "auto", None) else \
            ("cuda" if torch.cuda.is_available() else "cpu")
        step = max(1, int(getattr(self.model, "batch", 8)))
        chunks, ctx = [], windows
        def _np(v):
            return v.detach().float().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        # feature extraction is chunked: sampling-based models (Chronos-T5) blow up
        # a 6 GB card if every window is pushed through in one call
        for i in range(0, len(ctx), step):
            part = ctx[i:i + step]
            if grad:
                feats = self.model.classifier_features(part)         # adapter: differentiable
            else:
                with torch.no_grad():
                    feats = self.model.classifier_features(part)
            chunks.append(_np(feats).astype(np.float32))
        t = torch.as_tensor(np.concatenate(chunks, axis=0), device=dev)
        if not np.all(np.isfinite(t.detach().cpu().numpy() if hasattr(t, "detach") else t)):
            t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        return t

    # ------------------------------------------------------------------ native
    def fit_native(self, cds) -> TrainResult:
        t0 = time.time()
        mod = getattr(self.model, "classify_fit", None)
        if mod is None:
            raise CapabilityError(self.model.key, "clf_head", "no native classification head")
        X, y, tr, te = cds.parts()
        head = mod(list(X[tr]), y[tr])            # vendor builds + (optionally) fits it
        import torch
        if not isinstance(head, torch.nn.Module):   # vendor returned predictions
            return TrainResult(model=self.model.key, status="ok", steps=0,
                               seconds=time.time() - t0, extra={"route": "native",
                                                                "metrics": head})
        cfg = self.cfg
        n_cls = self.n_classes or cds.n_classes
        dev = "cuda" if torch.cuda.is_available() and cfg.device != "cpu" else "cpu"
        head = head.to(dev)
        params = [p for p in head.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=cfg.eff_lr())
        lossf = torch.nn.CrossEntropyLoss()
        log = StepLog(cfg.log_every)
        steps = cfg.eff_steps(max(8, int(tr.sum())))
        rng = np.random.RandomState(cfg.seed)
        head.train()
        Xtr, ytr = list(X[tr]), y[tr]
        for step in range(1, steps + 1):
            i = rng.choice(len(Xtr), size=min(cfg.bs, len(Xtr)), replace=False)
            xb = torch.tensor(np.stack([np.asarray(Xtr[j], float) for j in i]),
                              dtype=torch.float32, device=dev)[:, None, :]
            yb = torch.tensor(ytr[i], dtype=torch.long, device=dev)
            out = None
            if xb.shape[0] > 0:
                try:
                    out = head(xb)
                except TypeError:
                    # MOMENT's pipeline forward is keyword-only (x_enc=), not positional
                    out = head(x_enc=xb)
            logits = getattr(out, "class_predictions", None)
            if logits is None:
                logits = out if torch.is_tensor(out) else getattr(out, "logits", None)
            if logits is None:
                raise CapabilityError(self.model.key, "clf_head",
                                      "native head forward returned no logits")
            loss = lossf(logits[:, :n_cls], yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            log.step(step, float(loss.item()))
        head.eval()
        scores = []
        with torch.no_grad():
            for i in range(0, len(X), 16):
                chunk = np.stack([np.asarray(v, float) for v in X[i:i + 16]])
                xc = torch.tensor(chunk, dtype=torch.float32, device=dev)[:, None, :]
                try:
                    out = head(xc)
                except TypeError:
                    out = head(x_enc=xc)          # MOMENT's keyword-only pipeline forward
                lg = getattr(out, "class_predictions", None)
                lg = lg if lg is not None else (out if torch.is_tensor(out) else out.logits)
                scores.append(torch.softmax(lg[:, :n_cls], -1).cpu().numpy())
        rep = classification_report(y[te], np.concatenate(scores)[te], n_classes=n_cls)
        return TrainResult(model=self.model.key, status="ok", steps=steps,
                           first_loss=log.first, final_loss=log.last,
                           trainable_parameters=sum(p.numel() for p in params),
                           n_windows=int(tr.sum()), seconds=time.time() - t0,
                           extra={"route": "native", "metrics": rep})

    # ------------------------------------------------------------------ entry
    def fit(self, cds) -> TrainResult:
        """A route a family does not support returns ``skipped`` with a reason instead of
        raising: the matrix, the CLI and ordinary scripts all want the same verdict
        object, and "this model has no native head" is a fact, not an exception."""
        try:
            if self.route == "probe":
                return self.fit_probe(cds)
            if self.route == "native":
                return self.fit_native(cds)
            return self.fit_head(cds)
        except CapabilityError as e:
            return TrainResult(model=getattr(self.model, "key", "model"), status="skipped",
                               reason=str(e), n_windows=len(cds.X), seconds=0.0)
