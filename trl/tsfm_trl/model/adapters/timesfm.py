"""Google TimesFM 2.0 / 2.5 / 3.0.

* **2.5-200M** — the published survey case-study model. Inference through the pip
  package's compiled forecaster; fine-tuning trains the underlying
  ``TimesFM_2p5_200M_torch_module`` directly, reproducing the package's own
  two-stage normalisation (global REVIN over the context, then running-stat REVIN
  per patch) so training and inference agree. The output-patch layout (10 channels:
  0 = flip-invariance reference, 1..9 = levels 0.1…0.9, 5 = median) was verified
  empirically by predictive coverage, not guessed.
* **2.0-500M** — the v1 API (``TimesFm`` + ``TimesFmHparams``).
* **3.0** — multivariate (variate attention) with 9 quantiles; its weights carry the
  TimesFM non-commercial licence, and the public package ships inference only, so
  fine-tuning is reported as ``n/a(reason)`` rather than faked.
"""
from __future__ import annotations

import numpy as np

from ..base import LEVELS, Pred, TSFMModel, to_levels

POINT_CHANNEL = 5                     # median channel of the 2.5 head
QUANT_CHANNELS = [1, 2, 3, 4, 5, 6, 7, 8, 9]


class _TimesFMBase(TSFMModel):
    caps = frozenset({"forecast", "quantiles"})
    max_context = 512
    max_horizon = 128
    ctx_step = 32
    levels = list(LEVELS)


class TimesFM25(_TimesFMBase):
    key, family = "timesfm-2.5-200m", "timesfm25"
    caps = frozenset({"forecast", "quantiles", "finetune"})
    loss_kind = "quantile"
    head_prefixes = ("linear_output", "output_proj", "p21", "quantile")

    def __init__(self, checkpoint: str | None = None, *, per_core_batch_size: int = 16,
                 normalize_inputs: bool = True, torch_compile: bool = False, **kw):
        super().__init__(**kw)
        import timesfm
        self.checkpoint = checkpoint or timesfm.TimesFM_2p5_200M_torch.DEFAULT_REPO_ID
        self._wrapper = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            self.checkpoint, torch_compile=torch_compile)
        self._wrapper.compile(timesfm.ForecastConfig(
            max_context=self.max_context, max_horizon=self.max_horizon,
            normalize_inputs=normalize_inputs, infer_is_positive=False,
            force_flip_invariance=True, per_core_batch_size=per_core_batch_size,
            use_continuous_quantile_head=False, fix_quantile_crossing=True))
        self._norm = normalize_inputs

    # ------------------------------------------------------------------ inference
    def predict(self, contexts, horizon, *, panels=None, future=None) -> Pred:
        cs = self.trim(contexts)
        pts, qs = self._wrapper.forecast(horizon=int(horizon),
                                         inputs=[np.asarray(c, dtype=np.float64) for c in cs])
        pts = np.asarray(pts, float)[:, :horizon]
        Q = np.asarray(qs, float)[:, :horizon, :]
        q9 = Q[:, :, QUANT_CHANNELS] if Q.shape[-1] > 5 else Q
        return Pred(point=pts, quantiles=q9, meta={"channels": "1..9 = 0.1..0.9, 5 = median"})

    # ------------------------------------------------------------------ training
    def trainable(self):
        return self._wrapper.model

    def forward_quantiles(self, contexts, horizon):
        """Differentiable (B,H,9) quantile head in *global normalised* units — the
        space the 2.5 head is supervised in (see ``_patchify``)."""
        import torch
        outs = []
        for c in np.asarray(contexts, dtype=np.float32):
            normed, masks, cmu, csd = self._patchify(torch, c)
            module = self._wrapper.model
            (_, _, normed_outputs, _), _ = module(normed, masks, None)
            o, q = module.o, module.q
            last = torch.reshape(normed_outputs[:, -1, :], (1, o, q))
            from timesfm.torch import util as tfutil
            renorm = tfutil.revin(last, cmu[:, -1:], csd[:, -1:], reverse=True)
            outs.append(renorm[0, :horizon, QUANT_CHANNELS])
        return torch.stack(outs)

    def training_step(self, contexts, truth, horizon):
        """Per-sample step (the 2.5 head supervises its last output patch), averaged
        over the minibatch. Loss is computed in the head's own normalised space, in
        fp32 even under AMP, so the number stays comparable across precisions."""
        import torch
        from timesfm.torch import util as tfutil
        from ...training.losses import quantile_loss
        module = self._wrapper.model
        losses = []
        for c, y in zip(np.asarray(contexts, dtype=np.float32),
                        np.asarray(truth, dtype=np.float32)):
            normed, masks, cmu, csd = self._patchify(torch, c)
            (_, _, normed_outputs, _), _ = module(normed, masks, None)
            o, q = module.o, module.q
            last = torch.reshape(normed_outputs[:, -1, :], (1, o, q))
            renorm = tfutil.revin(last, cmu[:, -1:], csd[:, -1:], reverse=True).float()
            qh = renorm[:, : len(y), QUANT_CHANNELS]                    # (1,H,9)
            yt = self._target_in_norm(torch, c, y, module)              # (1,H)
            losses.append(quantile_loss(qh, yt, self.levels, w_point=1.0, w_cross=0.1))
        return torch.stack(losses).mean()

    # helpers ------------------------------------------------------------------
    def _patchify(self, torch, x, y=None):
        """Reproduce compile(normalize_inputs=True): global REVIN over the context,
        then running-stat REVIN per 32-step patch."""
        from timesfm.torch import util as tfutil
        module = self._wrapper.model
        p = module.p
        xt = torch.tensor(np.asarray(x, dtype=np.float32), device=self._dev()).unsqueeze(0)
        mu0 = xt.mean(dim=-1, keepdim=True)
        sd0 = xt.std(dim=-1, keepdim=True).clamp_min(1e-6)
        xin = (xt - mu0) / sd0 if self._norm else xt
        patched = torch.reshape(xin, (1, -1, p))
        masks = torch.zeros_like(patched, dtype=torch.bool)
        n = torch.zeros(1, device=patched.device)
        mu = torch.zeros(1, device=patched.device)
        sg = torch.zeros(1, device=patched.device)
        mus, sgs = [], []
        for i in range(patched.shape[1]):
            (n, mu, sg), _ = tfutil.update_running_stats(n, mu, sg, patched[:, i], masks[:, i])
            mus.append(mu)
            sgs.append(sg)
        cmu = torch.stack(mus, dim=1)
        csd = torch.stack(sgs, dim=1)
        normed = tfutil.revin(patched, cmu, csd, reverse=False)
        normed = torch.where(masks, 0.0, normed)
        return normed, masks, cmu, csd

    def _target_in_norm(self, torch, c, y, module):
        """Targets in the same globally-normalised units as the head output —
        without this the loss is off by the series scale."""
        xt = np.asarray(c, dtype=np.float32)
        mu0, sd0 = float(np.mean(xt)), max(float(np.std(xt)), 1e-6)
        yy = (np.asarray(y, dtype=np.float32) - mu0) / sd0 if self._norm else np.asarray(y, np.float32)
        return torch.tensor(yy, device=self._dev(), dtype=torch.float32).unsqueeze(0)

    def _dev(self):
        return next(self._wrapper.model.parameters()).device

    def load_finetuned_dir(self, path: str) -> dict:
        """A fine-tuned 2.5 saved by ``finetune``/``save_pretrained`` is reloaded as
        a full checkpoint directory (the vendor format), not as a state dict."""
        import timesfm
        self._wrapper = timesfm.TimesFM_2p5_200M_torch.from_pretrained(path, torch_compile=False)
        self._wrapper.compile(timesfm.ForecastConfig(
            max_context=self.max_context, max_horizon=self.max_horizon,
            normalize_inputs=self._norm, infer_is_positive=False, force_flip_invariance=True,
            per_core_batch_size=self.batch, use_continuous_quantile_head=False,
            fix_quantile_crossing=True))
        self.ft_load = {"via": "from_pretrained(dir)", "checkpoint": str(path)}
        return self.ft_load


class TimesFM20(_TimesFMBase):
    """TimesFM 2.0 (the *v1* python API, i.e. ``timesfm<2``).

    The package named ``timesfm==2.0.x`` on PyPI is the 2.5 release, not this model:
    only 1.2.x still ships ``TimesFm``/``TimesFmHparams`` (and it imports ``jax`` at
    package import, so that env carries jax even though we run the torch backend).
    The 500M checkpoint is 50 layers x 1280 dims x 16 heads -- building it with the
    2.5 shape loads a state_dict full of unexpected keys.
    """
    key, family = "timesfm-2.0-500m", "timesfm20"
    caps = frozenset({"forecast", "quantiles"})
    loss_kind = None
    max_context = 512
    max_horizon = 128

    #: what the released 2.0-500M checkpoint is
    arch = dict(num_layers=50, num_heads=16, model_dims=1280,
                input_patch_len=32, output_patch_len=128)
    levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def __init__(self, checkpoint: str = "google/timesfm-2.0-500m-pytorch", **kw):
        TSFMModel.__init__(self, **kw)
        import timesfm
        if not hasattr(timesfm, "TimesFmHparams"):
            from ...errors import UnavailableError
            raise UnavailableError(
                self.key,
                f"installed timesfm {getattr(timesfm, '__version__', '?')} has no v1 API "
                "(TimesFmHparams)",
                'this release needs "timesfm>=1.2.9,<2" (plus jax): use the timesfm20 env')
        self.checkpoint = checkpoint
        hp_kw = dict(self.arch)
        hp_kw.update(context_len=self.max_context, horizon_len=self.max_horizon,
                     per_core_batch_size=max(1, self.batch), backend="torch",
                     quantiles=list(self.levels))
        ck = timesfm.TimesFmCheckpoint(huggingface_repo_id=self.source_id or checkpoint)
        try:
            self._f = timesfm.TimesFm(hparams=timesfm.TimesFmHparams(**hp_kw),
                                      checkpoint=ck)
        except (TypeError, RuntimeError) as e:
            msg = str(e)
            if "state_dict" not in msg and "unexpected keyword" not in msg:
                raise
            # architecture mismatch: name the numbers instead of dumping 700 keys
            n = msg.count("self_attn.scaling")
            raise UnavailableError(
                self.key,
                f"checkpoint architecture does not match the built model "
                f"({n} extra layers in the state dict); expected {hp_kw['num_layers']} "
                f"layers / {hp_kw['model_dims']} dims",
                "pass checkpoint=<the 2.0 release that matches>, or set "
                "TimesFM20.arch from the checkpoint config") from e

    def predict(self, contexts, horizon, *, panels=None, future=None) -> Pred:
        cs = self.trim(contexts)
        pt, q = self._f.forecast(cs)
        pt = np.asarray(pt, float)[:, :horizon]
        q = np.asarray(q, float)[:, :horizon, :]
        use = None
        if q.ndim == 3 and q.shape[-1] >= 5:
            use = list(self.levels) if q.shape[-1] == len(self.levels) else \
                list(np.linspace(0.001, 0.999, q.shape[-1]))
            q = to_levels(q, use)      # -> canonical LEVELS grid on the standard axis
        return Pred(point=pt, quantiles=q if use else None,
                    meta={"levels": use, "normalisation": "TimesFM renorm off",
                          "backend": "torch (v1 API)"})

    def trainable(self):
        return None


class TimesFM30(_TimesFMBase):
    key, family = "timesfm-3.0", "timesfm30"
    caps = frozenset({"forecast", "quantiles", "multivariate", "covariates"})
    loss_kind = None
    max_context = 15360
    max_horizon = 512

    def __init__(self, checkpoint: str = "google/timesfm-3.0-pytorch", **kw):
        super().__init__(**kw)
        from timesfm import TimesFM3Forecaster
        self.checkpoint = checkpoint
        self.f = TimesFM3Forecaster.from_pretrained(checkpoint, device=self.device,
                                                    per_core_batch_size=self.batch)

    def predict(self, contexts, horizon, *, panels=None, future=None) -> Pred:
        cs = self.trim(contexts)
        outs = list(self.f.predict_batch(cs, horizon=int(horizon), return_quantiles=True))
        pts = np.stack([np.asarray(o.forecast, float)[-horizon:] for o in outs])
        ql = [getattr(o, "quantiles", None) for o in outs]
        qs = np.stack([np.asarray(q, float)[-horizon:] for q in ql]) if all(q is not None for q in ql) else None
        lv = list(getattr(self.f, "quantile_levels", None) or LEVELS)
        return Pred(point=pts, quantiles=qs, levels=lv,
                    meta={"variate_attention": True})

    def trainable(self):
        try:
            from timesfm import TimesFM3Torch
            return TimesFM3Torch.from_pretrained(self.checkpoint)
        except Exception:                                            # noqa: BLE001
            return None

    def training_step(self, contexts, truth, horizon):
        from ...errors import CapabilityError
        raise CapabilityError(
            self.key, "finetune",
            "the public timesfm package ships TimesFM 3.0 inference only (no documented "
            "training forward); contributing a head-only loop is tracked in docs/03_fine_tune.md")
