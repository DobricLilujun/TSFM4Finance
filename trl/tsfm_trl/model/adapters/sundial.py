"""THUML Sundial (Apache, ICML-2025 oral): a causal next-token model over
*time-series tokens*. Zero-shot inference goes through the model's own
``prepare_input_ids`` + token generation; the token vocabulary means fine-tuning
needs the vendor tokenizer ids, so without them the row is reported as
``finetune: n/a(reason)`` rather than trained with a made-up objective.
"""
from __future__ import annotations

import numpy as np

from ..base import Pred, TSFMModel, quantiles_from_samples, to_numpy


class Sundial(TSFMModel):
    """Sundial base-128m (THUML, ICML-2025 oral): continuous-value patch tokens + a
    TimeFlow (flow-matching) head.

    What the *current* repository revision actually exposes, verified on the matrix
    machine:

    * **no tokenizer files in the Hub repo and none needed** — ``generate`` takes the
      raw values: ``model.generate(torch.randn(B, lookback), max_new_tokens=H,
      num_samples=k)`` → ``(B, k, H)`` de-normalised samples (the mixin applies revin
      itself). ``prepare_input_ids``/``mode=``/``tokenizer=`` belong to an *older*
      revision that the model card's prose still mentions; the adapter supports both,
      preferring whichever the loaded code offers.
    * the Auto* path needs ``transformers==4.40.1`` (the version in the model card's own
      Quickstart; 4.42.x mis-builds the causal mask for patched inputs), and
      ``device_map=`` must not be used with this remote code (accelerate re-dispatches
      through the Auto mapping, which does not know ``model_type: sundial``).
    * **no harness-runnable supervised step**: ``forward(labels=...)`` reaches the
      vendor's TimeFlow loss, whose packing (``output_token_lens``,
      ``diffusion_batch_mul``, ``mask_y``) is set by their training scripts, not by the
      checkpoint config — calling it with the checkpoint's config fails on shape
      (`addmm: (8x17664) @ (768x768)`). Fine-tuning therefore reports ``—`` with that
      reason instead of optimising an invented objective. Their GitHub training script
      is the supported route.
    """

    key, family = "sundial-base-128m", "sundial"
    caps = frozenset({"forecast", "quantiles", "paths", "finetune"})
    # None, not "vendor": the objective is the checkpoint repo's own FlowLoss, but the
    # *loop* is ours (ladder, LoRA, provenance), so it goes through the generic path
    # rather than the vendor_fit escape hatch.
    loss_kind = None
    lora_trains_head = True
    max_context = 2880                       # the model card's Quickstart lookback
    max_horizon = 720                        # one TimeFlow token block
    head_prefixes = ("flow_loss", "embed_layer", "norm")

    def __init__(self, checkpoint: str = "thuml/sundial-base-128m", n_paths: int = 8, **kw):
        super().__init__(**kw)
        from transformers import AutoModelForCausalLM
        self.checkpoint = checkpoint
        self.n_paths = int(n_paths)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True)
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, contexts, horizon, *, panels=None, future=None) -> Pred:
        import torch
        horizon = int(horizon)
        x = torch.tensor(np.asarray(self.trim(contexts), dtype=np.float32),
                         device=self.device)
        kwargs = dict(max_new_tokens=max(horizon, 1), num_samples=max(1, self.n_paths))
        if hasattr(self.model, "prepare_input_ids"):        # older repo revision
            outs = []
            for c in self.trim(contexts):
                inputs, _ = self.model.prepare_input_ids(
                    X=np.asarray(c, dtype=np.float64)[:, None], normalize=True,
                    system="forecast", horizon=horizon)
                with torch.no_grad():
                    o = self.model.generate(inputs, mode="short_term_forecast",
                                            max_new_tokens=horizon, num_samples=self.n_paths)
                outs.append(np.asarray(o, dtype=float).reshape(self.n_paths, -1)[:, :horizon])
            S = np.stack(outs)
        else:
            with torch.no_grad():
                out = self.model.generate(x, **kwargs)      # (B, k, gen_len)
            S = to_numpy(out)[:, :, :horizon]
            if S.shape[-1] < horizon:
                # the head emits one block; extend it flat rather than pretending the
                # model predicted further ahead than it did
                S = np.concatenate([S, np.repeat(S[:, :, -1:], horizon - S.shape[-1],
                                                 axis=2)], axis=2)
        P = S.mean(axis=1)
        return Pred(point=P, quantiles=quantiles_from_samples(S), samples=S,
                    meta={"n_paths": S.shape[1], "how": "TimeFlow samples (revin-normalised)",
                          "api": "generate(values)"})

    def prepare_training(self, ctx: int, horizon: int) -> None:
        """Re-build the TimeFlow head at the *training* window length.

        The released checkpoint's ``output_token_lens=[720]`` is a generation setting:
        ``forward(labels=…)`` trims its labels to ``seq_len - input_token_len + 720``, a
        length an input window never reaches, so the shipped 720-wide flow MLP literally
        cannot supervise a short window. Their fine-tuning configures the output token
        length per run; we do the same, and say plainly what that costs: the head is
        **re-initialised** (~11.4M params) while the 12-layer trunk carries over — which
        is precisely the transfer a fine-tune of this family measures. The objective is
        ``flow_loss.FlowLoss`` from the checkpoint's own repo, not an invented loss.
        """
        import torch                                   # noqa: F401 (device handling)
        cfg = self.model.config
        self.t_out = int(getattr(cfg, "input_token_len", 16))
        cfg.output_token_lens = [self.t_out]
        fl = self._flow_loss_cls()
        self.model.flow_loss = fl(self.t_out, cfg.hidden_size, cfg.flow_loss_depth,
                                  cfg.hidden_size, cfg.num_sampling_steps).to(self.device)

    def _flow_loss_cls(self):
        import importlib
        import os
        import sys
        root = self.checkpoint if os.path.isdir(self.checkpoint) else None
        if root is None:
            from huggingface_hub import snapshot_download
            root = snapshot_download(self.checkpoint)
        if root not in sys.path:
            sys.path.insert(0, root)              # the repo's files import each other
        return importlib.import_module("flow_loss").FlowLoss

    def training_step(self, contexts, truth, horizon):
        """Teacher-forced next-patch TimeFlow loss, AR-aligned the way ``generate``
        behaves: the hidden state after patch *i* supervises patch *i+1*. revin uses the
        model's own statistics (mean/std over the window, std floored at 1e-2), which is
        what the mixin inverts at inference."""
        import torch
        cs = self.trim(contexts)
        y = np.asarray(truth, dtype=np.float32).reshape(len(cs), -1)
        window = np.concatenate([cs, y], axis=1)
        t = torch.tensor(window, dtype=torch.float32, device=self.device)
        means = t.mean(1, keepdim=True)
        stdev = t.std(1, keepdim=True, unbiased=False).clamp_min(1e-2)
        x = (t - means) / stdev
        T = getattr(self, "t_out", None) or int(getattr(self.model.config,
                                                        "input_token_len", 16))
        out = self.model.model(input_ids=x, use_cache=False, return_dict=True)
        hid = out.last_hidden_state                              # (B, n_patches, hidden)
        nxt = torch.cat([x[:, T:], torch.zeros(x.shape[0], T, device=self.device)], 1)
        lab = nxt.unfold(dimension=-1, size=T, step=T)           # (B, n, T)
        n = min(hid.shape[1], lab.shape[1])
        hid, lab = hid[:, :n], lab[:, :n]
        mask = torch.ones(hid.shape[0], n, device=self.device)
        mask[:, -1] = 0.0                       # nothing follows the last patch
        return self.model.flow_loss(lab.reshape(-1, T), hid.reshape(-1, hid.shape[-1]),
                                    mask.reshape(-1),
                                    torch.ones(1, T, device=self.device))

    def trainable(self):
        return self.model
