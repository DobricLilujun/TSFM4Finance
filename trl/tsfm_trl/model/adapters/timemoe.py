"""Time-MoE (Apache, ``transformers`` remote code).

The released class generates *greedy* continuous tokens, so a probabilistic row is
produced by sampling S paths with temperature and taking their quantiles — recorded
in ``Pred.meta``, because "how the interval was made" belongs in the result.
"""
from __future__ import annotations

import numpy as np

from ..base import Pred, TSFMModel, quantiles_from_samples


class TimeMoE(TSFMModel):
    key, family = "timemoe-50m", "timemoe"
    caps = frozenset({"forecast", "finetune", "paths"})
    loss_kind = "point"                  # value head -> Huber in normalised space
    max_context = 4096
    max_horizon = 512
    head_prefixes = ("lm_head", "model")

    def __init__(self, checkpoint: str = "Maple728/TimeMoE-50M", n_paths: int = 32, **kw):
        super().__init__(**kw)
        from transformers import AutoModelForCausalLM
        self.checkpoint = checkpoint
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.n_paths = int(n_paths)

    def _ar_decode(self, t0, horizon: int, *, sample: bool, temperature: float = 0.8,
                   top_p: float = 0.95):
        """Hand-rolled AR decode with KV cache.

        ``generate(do_sample=True)`` is *broken* in the released Time-MoE remote code:
        its ``prepare_inputs_for_generation`` hands transformers a 3-D tensor and
        ``_sample`` unpacks it as 2-D, so it raises for every transformers version we
        tested (4.38–4.44). Greedy generate works, but the probabilistic rows need
        sampled paths, so the loop is written here. The continuous head means a path is
        the predicted value plus temperature noise — Time-MoE's own sampling recipe, not
        an ad-hoc smoothing of a point forecast."""
        import torch
        # NOTE Time-MoE's forward *mutates the tensor you hand it* (it unsqueezes
        # input_ids in place), so never keep an alias to it: clone on the way in.
        out = t0.clone()
        past, step = None, t0.clone()
        for _ in range(int(horizon)):
            with torch.no_grad():
                o = self.model(input_ids=step, past_key_values=past, use_cache=True)
            past = o.past_key_values
            nxt = o.logits[:, -1, 0].float()
            if sample:
                nxt = nxt + temperature * torch.randn_like(nxt)
            step = nxt[:, None].clone()
            out = torch.cat([out, step], dim=1)
        return out[:, -int(horizon):]

    def predict(self, contexts, horizon, *, panels=None, future=None) -> Pred:
        import torch
        cs = self.trim(contexts)
        x = np.stack(cs)
        mean = x.mean(axis=1, keepdims=True)
        sd = np.where(x.std(axis=1, keepdims=True) > 0, x.std(axis=1, keepdims=True), 1.0)
        t0 = torch.tensor((x - mean) / sd, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            greedy = self._ar_decode(t0, horizon, sample=False).detach().float().cpu().numpy()
        # one batched decode for all paths: each window is repeated n_paths times so
        # the AR loop runs once, not once per path
        torch.manual_seed(self.seed)
        rep = t0.repeat_interleave(self.n_paths, dim=0)
        paths = self._ar_decode(rep, horizon, sample=True).detach().float().cpu().numpy()
        N = t0.shape[0]
        paths = paths.reshape(N, self.n_paths, -1) * sd[:, None, :] + mean[:, None, :]
        greedy = greedy * sd + mean
        return Pred(point=greedy, quantiles=quantiles_from_samples(paths), samples=paths,
                    meta={"n_paths": self.n_paths, "normalisation": "per-window mean/std",
                          "interval": "empirical from temperature-sampled paths",
                          "decode": "manual AR (vendor generate(do_sample=True) is broken)"})

    def sample_paths(self, contexts, horizon, n=8):
        """Normalised-space paths are fine for the GRPO reward (it is scale-free)."""
        p = self.predict(contexts, horizon)
        return np.asarray(p.samples, float)[:, :n, :horizon]

    def training_step(self, contexts, truth, horizon):
        from ._arutils import ar_value_step
        return ar_value_step(self.model, contexts, truth, horizon,
                             device=self.device, key=self.key)

    def trainable(self):
        return self.model
