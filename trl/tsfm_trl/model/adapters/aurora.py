"""Aurora (MIT, ICLR-2026): multimodal TSFM — modality-guided attention plus
prototype-guided **flow matching**, so it emits generative samples rather than a
quantile grid.

    pip install aurora-model          # or trust_remote_code from DecisionIntelligence/Aurora
    m = load_model()
    o = m.generate(inputs=seqs, max_output_length=96, num_samples=20, inference_token_len=48)

Because the head is a flow field, the training objective is not pinball; the
library keeps it inference-only until a flow-matching step is wired (see
docs/05_reward_rl.md), and the matrix says so.
"""
from __future__ import annotations

import numpy as np

from ..base import Pred, TSFMModel, quantiles_from_samples


class Aurora(TSFMModel):
    key, family = "aurora", "aurora"
    caps = frozenset({"forecast", "paths"})
    loss_kind = None
    max_context = 512
    max_horizon = 96
    n_samples = 20

    def __init__(self, checkpoint: str = "DecisionIntelligence/Aurora", num_samples: int = 20,
                 **kw):
        super().__init__(**kw)
        self.checkpoint, self.num_samples = checkpoint, int(num_samples)
        self.model = self._load(checkpoint)
        # the vendor's load_model() hands back a model in train mode, and its norm
        # layers then raise "Expected more than 1 value per channel" on the batch of
        # one that inference runs are made of
        if hasattr(self.model, "eval"):
            self.model.eval()

    @staticmethod
    def _load(checkpoint: str):
        try:
            from aurora import load_model
            return load_model()
        except Exception:                                          # noqa: BLE001
            from transformers import AutoModel
            return AutoModel.from_pretrained(checkpoint, trust_remote_code=True)

    def predict(self, contexts, horizon, *, panels=None, future=None) -> Pred:
        import torch
        cs = self.trim(contexts)
        paths = []
        for c in cs:
            x = torch.tensor(np.asarray(c, dtype=np.float32)[None, :], device=self.device)
            with torch.no_grad():
                out = self.model.generate(inputs=x, max_output_length=int(horizon),
                                          num_samples=self.num_samples,
                                          inference_token_len=min(48, len(c)))
            a = np.asarray(out.cpu() if hasattr(out, "cpu") else out, float)
            a = a.reshape(self.num_samples, -1)[:, :horizon]
            paths.append(a)
        S = np.stack(paths)
        med = np.median(S, axis=1)
        return Pred(point=med, quantiles=quantiles_from_samples(S), samples=S,
                    meta={"head": "flow matching (generative)", "num_samples": self.num_samples})
