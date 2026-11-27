"""Kairos (Apache, ``mldi-lab/Kairos_*``): encoder-decoder TSFM whose tokenization
decouples time-series heterogeneity from model capacity.

Loaded through ``transformers`` (the repos declare ``model_type: kairos``), with a
fallback to the authors' ``tsfm.model.kairos.AutoModel`` when the installed
transformers does not know the class. The forward follows HF time-series
conventions: ``model(past_target=…, prediction_length=…)`` → a dict with
``prediction_outputs``.
"""
from __future__ import annotations

import inspect

import numpy as np

from ...errors import CapabilityError, UnavailableError
from ..base import LEVELS, Pred, TSFMModel, to_levels


class Kairos(TSFMModel):
    key, family = "kairos-10m", "kairos"
    caps = frozenset({"forecast", "quantiles", "finetune"})
    loss_kind = "quantile"
    max_context = 2048
    max_horizon = 128
    ctx_step = 128
    head_prefixes = ("decoder", "projection", "head", "p21")

    def __init__(self, checkpoint: str = "mldi-lab/Kairos_10m", **kw):
        super().__init__(**kw)
        self.checkpoint = checkpoint
        self.model = self._load(checkpoint)
        try:
            self.model = self.model.to(self.device)
        except Exception:                                          # noqa: BLE001
            pass
        self.model.eval()
        cfg = getattr(self.model, "config", None)
        self.levels = [float(x) for x in (getattr(cfg, "quantiles", None) or [])] or None
        self.pred_len = int(getattr(cfg, "prediction_length", 64) or 64)
        self.max_horizon = max(self.pred_len, 64)
        self._fwd_params = set(inspect.signature(self.model.forward).parameters)

    @staticmethod
    def _load(checkpoint: str):
        from transformers import AutoModel
        try:
            return AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
        except Exception as e1:                                    # noqa: BLE001
            try:
                from tsfm.model.kairos import AutoModel as KA
                return KA.from_pretrained(checkpoint, trust_remote_code=True)
            except Exception:                                      # noqa: BLE001
                raise UnavailableError(
                    "kairos", f"transformers does not know model_type 'kairos' and the "
                              f"authors' package is not importable ({e1})",
                              "pip install git+https://github.com/foundation-model-research/Kairos") from e1

    def _reshape_out(self, raw, horizon):
        """Heuristic shape normalisation for ``prediction_outputs``."""
        import torch
        a = raw.detach().float() if isinstance(raw, torch.Tensor) else np.asarray(raw, float)
        nq = len(self.levels) if self.levels else 1
        if hasattr(a, "cpu"):
            a_np = a.cpu().numpy()
        else:
            a_np = a
        a_np = np.asarray(a_np, float)
        if a_np.ndim == 2:                                   # (B,H)
            return a_np[:, :horizon, None], None
        if a_np.ndim == 4:                                   # (B,levels,H,1) style
            a_np = a_np[:, :, :, 0]
        if a_np.ndim == 3:
            if a_np.shape[-1] == nq:
                return a_np[:, :horizon, :], None
            if a_np.shape[1] == nq:
                return a_np[:, :horizon, :].transpose(0, 2, 1), None
            if a_np.shape[0] == nq:
                return np.transpose(a_np, (1, 2, 0))[:, :horizon, :], None
        # last resort: (B, levels*H)
        if a_np.ndim == 2 and nq > 1 and a_np.shape[1] % nq == 0:
            H = a_np.shape[1] // nq
            return a_np.reshape(a_np.shape[0], H, nq)[:, :horizon, :], None
        raise UnavailableError(self.key, f"unrecognised prediction_outputs shape {a_np.shape}")

    def predict(self, contexts, horizon, *, panels=None, future=None) -> Pred:
        import torch
        cs = self.trim(contexts)
        x = torch.tensor(np.stack(cs), dtype=torch.float32, device=self.device)
        kw = {"past_target": x, "prediction_length": int(horizon)}
        if "generation" in self._fwd_params:
            kw["generation"] = False
        with torch.no_grad():
            out = self.model(**kw)
        raw = out["prediction_outputs"] if isinstance(out, dict) else \
            getattr(out, "prediction_outputs", None)
        if raw is None:
            raise UnavailableError(self.key, "forward returned no prediction_outputs")
        q, _ = self._reshape_out(raw, horizon)
        if q.shape[1] != horizon:
            # some Kairos releases ignore prediction_length and emit their configured
            # horizon; slice it rather than returning a (n, 128) array under a
            # request for 16, which the matrix would (rightly) call bad-shape
            q = q[:, :horizon, :] if q.shape[1] > horizon else q
            meta_note = f"model emitted {q.shape[1]} steps for a {horizon}-step request"
        else:
            meta_note = ""
        nq = q.shape[-1]
        lv = (self.levels or [])[:nq] if self.levels and len(self.levels) >= nq else \
            [0.1 * (i + 1) for i in range(nq)]
        med = int(np.argmin(np.abs(np.asarray(lv) - 0.5)))
        return Pred(point=q[:, :, med], quantiles=(to_levels(q, lv) if nq > 1 else None),
                    levels=(LEVELS if nq > 1 else None),
                    meta={"horizon_note": meta_note} if meta_note else {})

    def training_step(self, contexts, truth, horizon):
        """Try the model's own supervised forward (``future_target`` / ``labels``);
        if the installed checkpoint only exposes the generation path, say so."""
        import torch
        cs = self.trim(contexts)
        x = torch.tensor(np.asarray(cs), dtype=torch.float32, device=self.device)
        y = torch.tensor(np.asarray(truth), dtype=torch.float32, device=self.device)
        for kwname in ("future_target", "labels", "target"):
            if kwname in self._fwd_params:
                out = self.model(past_target=x, prediction_length=int(y.shape[1]),
                                 **{kwname: y})
                loss = out.get("loss") if isinstance(out, dict) else getattr(out, "loss", None)
                if loss is not None:
                    # their supervised forward returns one loss per window; the optimiser
                    # needs a scalar, and the mean is what their own loop minimises
                    return loss.mean() if loss.ndim else loss
        raise CapabilityError(self.key, "finetune",
                              f"forward({sorted(self._fwd_params)}) exposes no supervised "
                              f"target/loss path (a differentiable head was not found); a "
                              f"Kairos fine-tune needs the authors' training loop — refusing "
                              f"to optimise a non-differentiable surrogate")

    def trainable(self):
        return self.model
