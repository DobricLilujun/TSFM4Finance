"""Time-MoE (Apache, transformers remote code).  The released class generates
*greedy* continuous tokens only, so probabilistic rows come from sampling S paths
with temperature; that choice is recorded in Pred.meta so the table is honest about
how the interval was produced."""
from __future__ import annotations

import numpy as np
import torch

from .base import Engine, Pred, quantiles_from_samples


class TimeMoEEngine(Engine):
    key, family = "timemoe-50m", "timemoe"
    caps = {"forecast", "finetune"}
    max_context = 4096
    max_horizon = 512

    def __init__(self, checkpoint: str = "Maple728/TimeMoE-50M", n_paths: int = 32, **kw):
        super().__init__(**kw)
        from transformers import AutoModelForCausalLM
        self.checkpoint = checkpoint
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.n_paths = int(n_paths)
        self.key = self.key.replace("50m", checkpoint.split("-")[-1].lower())

    def predict(self, contexts, horizon, *, panels=None) -> Pred:
        cs = self.trim(contexts)
        x = np.stack(cs)
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True)
        std = np.where(std > 0, std, 1.0)
        normed = (x - mean) / std
        t0 = torch.tensor(normed, dtype=torch.float32, device=self.device)
        outs = []
        with torch.no_grad():
            for k in range(self.n_paths):
                if k == 0:
                    o = self.model.generate(t0, max_new_tokens=horizon, do_sample=False)
                else:
                    g = torch.Generator(device=self.device).manual_seed(self.seed + k)
                    o = self.model.generate(t0, max_new_tokens=horizon, do_sample=True,
                                            temperature=0.8, top_p=0.95, generator=g)
                outs.append(np.asarray(o[:, -horizon:].detach().float().cpu()))
        paths = np.stack(outs, axis=1) * std[:, None, :] + mean[:, None, :]   # (N,S,H)
        greedy = np.asarray(self.model.generate(t0, max_new_tokens=horizon,
                                                do_sample=False)[:, -horizon:]
                            .detach().float().cpu()) * std + mean
        return Pred(point=greedy, quantiles=quantiles_from_samples(paths), samples=paths,
                    meta={"n_paths": self.n_paths, "normalization": "per-window mean/std"})

    def trainable(self):
        return self.model
