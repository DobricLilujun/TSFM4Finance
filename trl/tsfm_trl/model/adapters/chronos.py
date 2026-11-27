"""Amazon Chronos family: Chronos-2 (in-context group attention), Chronos-Bolt
(patch-based quantile head), Chronos-T5 (discrete-token autoregressive).

All three emit a quantile grid; each adapter maps its own grid onto the shared
``LEVELS`` so CRPS/WQL/VaR are comparable across models.

Fine-tuning paths, chosen to match each family's own training semantics:

* **Chronos-2** — the vendor's ``Chronos2Pipeline.fit`` (or the model's own
  ``forward(future_target=…)`` loss). Reimplementing that loop would quietly change
  what "fine-tuned Chronos-2" means.
* **Chronos-Bolt** — the model's internal quantile loss in its scaled space.
* **Chronos-T5** — next-token cross-entropy on the model's own tokenizer, which
  also gives sampled paths and per-path log-probs for the reward/GRPO trainer.
"""
from __future__ import annotations

import numpy as np

from ...errors import UnavailableError
from ..base import LEVELS, Pred, TSFMModel, to_levels


class _ChronosBase(TSFMModel):
    caps = frozenset({"forecast", "quantiles", "finetune"})
    max_context = 2048
    max_horizon = 256

    def _discover_levels(self) -> list[float]:
        for holder in (getattr(self, "pipe", None), getattr(self.pipe, "model", None),
                       getattr(getattr(self.pipe, "model", None), "config", None)):
            for attr in ("quantiles", "quantile_levels"):
                v = getattr(holder, attr, None) if holder is not None else None
                if v is not None:
                    try:
                        out = [float(x) for x in (v.tolist() if hasattr(v, "tolist") else list(v))]
                        if len(out) >= 3:
                            return out
                    except Exception:                              # noqa: BLE001
                        pass
        return list(LEVELS)

    def trainable(self):
        """The HF/Chronos ``nn.Module`` behind the pipeline — what the generic
        trainer and PEFT operate on."""
        return getattr(self, "pipe", None) and self.pipe.model

    def _t(self, arr, dtype=None):
        import torch
        return torch.tensor(np.asarray(arr, dtype=np.float32),
                            dtype=dtype or torch.float32, device=self.device)

    def _normalise_q(self, q):
        """Chronos shapes differ per generation: reduce to (N,H,nq)."""
        q = np.asarray(q, float)
        while q.ndim > 3:
            q = q[:, 0]
        if q.ndim == 2:
            return q[:, :, None]
        n_lv = len(self.levels)
        if q.shape[1] == n_lv:
            q = q.transpose(0, 2, 1)                          # (N,nq,H) -> (N,H,nq)
        return q


class Chronos2(_ChronosBase):
    key, family = "chronos-2", "chronos2"
    caps = frozenset({"forecast", "quantiles", "covariates", "multivariate", "finetune"})
    loss_kind = "vendor"
    ctx_step = 8

    def __init__(self, checkpoint: str = "amazon/chronos-2", **kw):
        super().__init__(**kw)
        import torch
        from chronos import Chronos2Pipeline
        self.checkpoint = checkpoint
        dtype = torch.bfloat16 if self.amp in ("bf16", "fp16") else torch.float32
        self.pipe = Chronos2Pipeline.from_pretrained(checkpoint, device_map=self.device, dtype=dtype)
        self.levels = self._discover_levels()

    def predict(self, contexts, horizon, *, panels=None, future=None) -> Pred:
        import torch
        cs = self.trim(contexts)
        inp = [torch.tensor(c, dtype=torch.float32) for c in cs]
        kw = {}
        if future is not None:
            kw["future_covariates"] = [torch.tensor(np.asarray(f), dtype=torch.float32)
                                       for f in future]
        q = self.pipe.predict(inp, prediction_length=int(horizon), **kw)
        q = np.asarray(q.detach().cpu() if isinstance(q, torch.Tensor) else q, float)
        q = self._normalise_q(q)[:, :horizon, :]
        qs = to_levels(q, self.levels[: q.shape[-1]])
        med = int(np.argmin(np.abs(np.asarray(self.levels) - 0.5)))
        return Pred(point=q[:, :, min(med, q.shape[-1] - 1)], quantiles=qs, levels=LEVELS)

    def encode(self, contexts):
        """Group-attention embeddings (mean pooled) for the probe route."""
        import torch
        cs = self.trim(contexts)
        x = torch.tensor(np.stack(cs)[:, None, :], dtype=torch.float32, device=self.device)
        m = self.pipe.model
        with torch.no_grad():
            try:
                out = m(context=x, output_hidden_states=True)
                hs = getattr(out, "hidden_states", None)
                emb = hs[-1] if hs is not None else getattr(out, "last_hidden_state", None)
                if emb is None:
                    raise AttributeError("no hidden states")
                return np.asarray(emb.float().mean(dim=-2).cpu(), float)
            except Exception:                                      # noqa: BLE001
                return TSFMModel.encode(self, contexts)

    def training_step(self, contexts, truth, horizon):
        """Chronos-2's own supervised forward, with the output length set per run.

        ``num_output_patches`` is a **forward argument** in chronos ≥2.3 and the vendor
        derives it from the prediction length
        (``num_output_patches * output_patch_size >= future_length``). Its default of 1
        caps a supervised window at one output patch — 16 steps — so a 21-step horizon
        died inside ``_validate_input`` with an error that named neither the argument nor
        the cap, and the channels-last retry below turned that into a still more
        misleading "context must have shape (batch, context_length)". Deriving it here is
        what their own fine-tuning does; nothing about the checkpoint is rebuilt, the
        output embedding is applied per patch.
        """
        import math
        import torch
        m = self.pipe.model
        x = torch.tensor(np.asarray(contexts, dtype=np.float32), device=self.device)
        ft = torch.tensor(np.asarray(truth, dtype=np.float32), device=self.device)
        if ft.ndim == 1:
            ft = ft[None, :]
        ops = int(getattr(getattr(m, "chronos_config", None), "output_patch_size", 0) or 0)
        kw = {"num_output_patches": max(1, math.ceil(ft.shape[1] / ops))} if ops else {}
        try:
            out = m(context=x, future_target=ft, **kw)
        except (ValueError, RuntimeError) as first:
            try:                        # some chronos builds want (B, T, 1) channels-last
                out = m(context=x.unsqueeze(-1), future_target=ft.unsqueeze(-1), **kw)
            except Exception:
                raise first             # the first error is the one worth reporting
        loss = getattr(out, "loss", None)
        if loss is None:
            raise RuntimeError(f"Chronos2Model returned no loss for future_target "
                               f"(horizon {ft.shape[1]}, {kw or 'default output patches'})")
        return loss.mean() if loss.ndim else loss

    def vendor_fit(self, windows, cfg):
        """The vendor loop: each sample is handed over as one series of exactly
        ctx+horizon points (``fit`` windows the history itself)."""
        import torch
        from chronos import Chronos2Pipeline
        from ...weights import resolve
        inputs = [{"target": torch.tensor(np.r_[w[0], w[1]], dtype=torch.float32)} for w in windows]
        # their Trainer reports the curve through callbacks, so a long run's losses are
        # captured instead of lost to stdout. At smoke size (a handful of steps) their
        # default logging interval never fires, so a smoke cell stays a bare ✅ with no
        # curve — recorded, not hidden: see the note in docs/03_fine_tune.md.
        losses: list[float] = []
        try:
            from transformers.trainer_callback import TrainerCallback
        except Exception:                                          # noqa: BLE001
            TrainerCallback = object

        class _Collect(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kw):
                v = (logs or {}).get("loss")
                if v is not None:
                    losses.append(float(v))

        steps = 2 if cfg.smoke else int(cfg.eff_steps(len(windows)))
        pipe = Chronos2Pipeline.from_pretrained(resolve(self.checkpoint) or self.checkpoint,
                                                device_map=cfg.device if cfg.device != "auto"
                                                else self.device, dtype=torch.float32)
        out_dir = str(cfg.ckpt_path(self.key).parent / f"{self.key}__{cfg.budget}_pipe")
        ft = pipe.fit(inputs, prediction_length=cfg.horizon,
                      finetune_mode=("full" if cfg.peft == "full" else "lora"),
                      context_length=cfg.ctx, learning_rate=cfg.eff_lr(), num_steps=steps,
                      batch_size=cfg.bs, output_dir=out_dir, min_past=min(cfg.ctx, 32),
                      callbacks=[_Collect()])
        state = ft.model.state_dict()
        torch.save({"model": state,
                    "meta": {"model": self.key, "vendor_fit": True, "steps": steps,
                             "peft": cfg.peft, "budget": cfg.budget, "smoke": cfg.smoke,
                             "trainable_parameters": sum(p.numel() for p in ft.model.parameters())}},
                   cfg.ckpt_path(self.key))
        # their HF Trainer keeps its own log; surface it, or the matrix cell shows a
        # ✅ with no curve and a diverging vendor run would be invisible
        ls = list(losses)
        return {"checkpoint": str(cfg.ckpt_path(self.key)), "steps": steps,
                "trainable_parameters": sum(p.numel() for p in ft.model.parameters()),
                "first_loss": ls[0] if ls else float("nan"),
                "final_loss": ls[-1] if ls else float("nan"),
                "vendor_train_loss": ls[-1] if ls else float("nan")}

    def load_finetuned(self, path):
        """Chronos-2 fine-tunes are commonly kept as a whole pipeline directory."""
        from pathlib import Path
        p = Path(path)
        if p.is_dir():
            from chronos import Chronos2Pipeline
            import torch
            self.pipe = Chronos2Pipeline.from_pretrained(str(p), device_map=self.device,
                                                         dtype=torch.float32)
            self.ft_load = {"via": "pipeline dir", "checkpoint": str(p)}
            self.ft_meta = {}
            return self
        from ...training.checkpoints import load_finetuned as _load_sd
        return _load_sd(self, path)


class ChronosBolt(_ChronosBase):
    key, family = "chronos-bolt", "chronos_bolt"
    loss_kind = "quantile"
    max_horizon = 64
    head_prefixes = ("output_patch_embedding", "quantiles")

    def __init__(self, checkpoint: str = "amazon/chronos-bolt-small", **kw):
        super().__init__(**kw)
        import torch
        from chronos import ChronosBoltPipeline
        self.checkpoint = checkpoint
        dtype = torch.bfloat16 if self.amp in ("bf16", "fp16") else torch.float32
        self.pipe = ChronosBoltPipeline.from_pretrained(checkpoint, device_map=self.device,
                                                        dtype=dtype)
        self.levels = self._discover_levels()

    def predict(self, contexts, horizon, *, panels=None, future=None) -> Pred:
        import torch
        cs = self.trim(contexts)
        q = self.pipe.predict([torch.tensor(c, dtype=torch.float32) for c in cs],
                              prediction_length=int(horizon))
        q = np.asarray(q.detach().cpu() if isinstance(q, torch.Tensor) else q, float)
        q = self._normalise_q(q)[:, :horizon, :]
        lv = self.levels[: q.shape[-1]] if len(self.levels) == q.shape[-1] else \
            list(np.linspace(0.05, 0.95, q.shape[-1]))
        med = int(np.argmin(np.abs(np.asarray(lv) - 0.5)))
        return Pred(point=q[:, :, med], quantiles=to_levels(q, lv))

    def forward_quantiles(self, contexts, horizon):
        """Differentiable quantile head in real units (the model's instance norm is
        inverted inside ``forward``)."""
        import torch
        x = torch.tensor(np.asarray(contexts, dtype=np.float32), device=self.device)
        return self.pipe.model(context=x).quantile_preds[..., :horizon].transpose(1, 2)

    def training_step(self, contexts, truth, horizon):
        import torch
        x = torch.tensor(np.asarray(contexts, dtype=np.float32), device=self.device)
        y = torch.tensor(np.asarray(truth, dtype=np.float32), device=self.device)
        out = self.pipe.model(context=x, target=y)
        loss = getattr(out, "loss", None)
        if loss is None:
            q = out.quantile_preds.transpose(1, 2)[..., : y.shape[1], :]
            from ...training.losses import quantile_loss
            loss = quantile_loss(q, y, self.levels[: q.shape[-1]])
        return loss


class ChronosT5(_ChronosBase):
    key, family = "chronos-t5", "chronos_t5"
    caps = frozenset({"forecast", "quantiles", "finetune", "paths"})
    loss_kind = "token"
    max_context = 512
    max_horizon = 64
    head_prefixes = ("lm_head", "shared", "encoder", "decoder")

    def __init__(self, checkpoint: str = "amazon/chronos-t5-small", n_paths: int = 32, **kw):
        super().__init__(**kw)
        from chronos import ChronosPipeline
        self.checkpoint = checkpoint
        self.pipe = ChronosPipeline.from_pretrained(checkpoint, device_map=self.device,
                                                    torch_dtype=None)
        self.cm = self.pipe.model
        # chronos-forecasting moved the tokenizer between releases (pipeline.tokenizer
        # in 2.x, model.input_tokenizer in 1.x); the token objective must use whichever
        # one *this* install uses, or CE training silently optimises the wrong ids.
        self.tok = getattr(self.pipe, "tokenizer", None) or getattr(self.cm, "input_tokenizer", None)
        if self.tok is None:
            raise UnavailableError(self.key, "this chronos build exposes no tokenizer; "
                                   "token cross-entropy training is unavailable")
        # NOTE: the tokenizer deliberately stays on the CPU. Its bin boundaries are
        # plain tensor attributes (not buffers, in every release we tested) and its
        # EOS helper allocates on the CPU, so _tokenize() tokenises on CPU and moves
        # the ids — the same thing the vendor pipeline does.
        self.n_paths = int(n_paths)
        self.levels = list(np.arange(0.1, 0.95, 0.1))

    def predict(self, contexts, horizon, *, panels=None, future=None) -> Pred:
        import torch
        cs = self.trim(contexts)
        q, samples = self.pipe.predict_quantiles([torch.tensor(c, dtype=torch.float32) for c in cs],
                                                 prediction_length=int(horizon),
                                                 num_samples=self.n_paths)
        q = np.asarray(q.detach().cpu() if isinstance(q, torch.Tensor) else q, float)
        q = self._normalise_q(q)[:, :horizon, :]
        s = np.asarray(samples.detach().cpu() if isinstance(samples, torch.Tensor) else samples,
                       float)[:, :horizon] if samples is not None else None
        lv = [float(x) for x in (getattr(self.cm.config, "quantile_levels", None) or self.levels)]
        from ..base import quantiles_from_samples
        out = Pred(point=to_levels(q, lv)[..., int(np.argmin(np.abs(np.asarray(lv) - 0.5)))],
                   quantiles=to_levels(q, lv))
        if s is not None and s.ndim == 3:
            out.samples = s
            out.quantiles = quantiles_from_samples(s)
        return out

    # ---- token plumbing for CE training + GRPO ----------------------------
    def _tokenize(self, contexts, truth=None):
        """Tokenise on CPU then move the ids: chronos' own ``_append_eos_token``
        allocates its EOS tensor on the CPU regardless of the input device, so
        tokenising straight on CUDA raises a device-mismatch error. The pipeline
        does the same dance internally."""
        import torch
        dev = torch.device(self.device)
        ctx = torch.tensor(np.asarray(contexts, dtype=np.float32))
        c_ids, c_mask, state = self.tok.context_input_transform(ctx)
        c_ids, c_mask = c_ids.to(dev), c_mask.to(dev)
        if truth is None:
            return c_ids, c_mask, state, None, None
        L = int(self.tok.config.prediction_length)
        t = np.asarray(truth, dtype=np.float32)
        if t.ndim == 1:
            t = t[None, :]
        if t.shape[1] != L:
            # Chronos-T5's label window is fixed by its config: optimise the first L
            # steps (the model's own native horizon) and record the truncation, rather
            # than pretending a 96-step head exists
            t = t[:, :L] if t.shape[1] > L else                 np.concatenate([t, np.repeat(t[:, -1:], L - t.shape[1], axis=1)], axis=1)
            self.ft_meta["label_window"] = f"{np.asarray(truth).shape[-1]}->{L}"
        tgt = torch.tensor(t)
        l_ids, l_mask = self.tok.label_input_transform(tgt, state)
        return c_ids, c_mask, state, l_ids.to(dev), l_mask.to(dev)

    def training_step(self, contexts, truth, horizon):
        c_ids, c_mask, _state, l_ids, _l_mask = self._tokenize(contexts, truth)
        out = self.cm.model(input_ids=c_ids, attention_mask=c_mask, labels=l_ids)
        loss = getattr(out, "loss", None)
        if loss is None:
            raise RuntimeError("Chronos-T5 returned no loss for labels")
        return loss

    def sample_paths(self, contexts, horizon, n=8):
        import torch
        c_ids, c_mask, state, *_ = self._tokenize(contexts)
        preds = self.cm(input_ids=c_ids, attention_mask=c_mask, prediction_length=int(horizon),
                        num_samples=int(n), temperature=1.0, top_p=0.95)
        vals = self.tok.output_transform(preds.cpu(), state)   # see _tokenize: CPU tokenizer
        v = np.asarray(vals.detach().cpu() if isinstance(vals, torch.Tensor) else vals, float)
        if v.ndim == 2:                                   # (B,H) when n == 1
            v = v[:, None, :]
        return v[:, :n, :horizon]

    def path_logprob(self, contexts, path):
        """Sum of next-token log-probs of ``path`` (per batch row). Used by the
        GRPO-style reward trainer; the same tokenizer as training, so the objective
        is the model's own likelihood."""
        import torch.nn.functional as F
        c_ids, c_mask, _s, l_ids, _lm = self._tokenize(contexts, path)
        out = self.cm.model(input_ids=c_ids, attention_mask=c_mask, labels=l_ids)
        logits = getattr(out, "logits", None)
        if logits is None:
            raise RuntimeError("Chronos-T5 forward returned no logits")
        lp = F.log_softmax(logits.float(), dim=-1)
        idx = l_ids.unsqueeze(-1)
        tok = lp.gather(-1, idx).squeeze(-1)
        return tok.sum(dim=-1)   # tensor: GRPO differentiates through this
