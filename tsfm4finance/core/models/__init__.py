"""Model adapters for Finance TSFMs Arena.

Each adapter wraps a concrete time-series model behind the unified BaseModel
contract. Adapters that require heavy dependencies (torch/huggingface) degrade
gracefully: if the dependency is missing, registering the adapter raises a
clear error but the rest of the arena still works.

Built-in models:
  - Baselines (no heavy deps): MovingAverage, SeasonalNaive, Naive, Random
  - Statistical: ARIMA (pmdarima)
  - TSFMs: Chronos (amazon), TimesFM (google), Moirai / UniTime (if installed)
"""
from __future__ import annotations

import abc
from typing import Optional

import numpy as np
import pandas as pd

from tsfm4finance.core.model_base import BaseModel, PredictionResult
from tsfm4finance.core.schemas import DatasetMeta, TaskType


def _get_series(context_df: pd.DataFrame, meta: DatasetMeta) -> pd.Series:
    """Return the target column as a Series (robust to missing target name)."""
    if meta.target in context_df.columns:
        return context_df[meta.target].astype(float)
    # fallback: last numeric column
    num = context_df.select_dtypes(include="number")
    if len(num.columns) == 0:
        raise ValueError(f"No numeric column / target {meta.target!r} in context_df")
    return num.iloc[:, -1].astype(float)


# --------------------------------------------------------------------------- #
# Baselines (always available)
# --------------------------------------------------------------------------- #
class MovingAverageModel(BaseModel):
    def __init__(self, name="moving_average", window: int = 10):
        super().__init__(name, {"forecast"})
        self.window = window

    def predict(self, context_df, meta, horizon=None):
        horizon = horizon or meta.horizon
        series = _get_series(context_df, meta).to_numpy()
        tail = series[-self.window:]
        base = float(np.mean(tail))
        return PredictionResult(point=[base] * horizon)


def _local_weights(hf_id: str) -> str:
    """Offline support: use ``$TSFM_WEIGHTS_ROOT/<org>/<repo>`` when it exists."""
    import os
    from pathlib import Path
    root = Path(os.environ.get("TSFM_WEIGHTS_ROOT")
                or Path(__file__).resolve().parents[3] / "models")
    for cand in (root / hf_id, root / hf_id.replace("/", "__"), root / hf_id.split("/")[-1]):
        if cand.is_dir() and any(f.suffix in (".json", ".safetensors", ".bin", ".pt", ".ckpt") for f in cand.iterdir()):
            return str(cand)
    return hf_id



class SeasonalNaiveModel(BaseModel):
    """Repeat the last season (for daily data, season=seasonality)."""

    def __init__(self, name="seasonal_naive", seasonality: int = 5):
        super().__init__(name, {"forecast"})
        self.seasonality = seasonality

    def predict(self, context_df, meta, horizon=None):
        horizon = horizon or meta.horizon
        series = _get_series(context_df, meta).to_numpy()
        s = self.seasonality
        out = []
        for h in range(horizon):
            idx = len(series) - 1 - (h % s)
            out.append(float(series[idx]))
        return PredictionResult(point=out)


class NaiveModel(BaseModel):
    def __init__(self, name="naive"):
        super().__init__(name, {"forecast"})

    def predict(self, context_df, meta, horizon=None):
        horizon = horizon or meta.horizon
        series = _get_series(context_df, meta).to_numpy()
        last = float(series[-1])
        return PredictionResult(point=[last] * horizon)


class RandomWalkModel(BaseModel):
    def __init__(self, name="random_walk"):
        super().__init__(name, {"forecast"})

    def predict(self, context_df, meta, horizon=None):
        horizon = horizon or meta.horizon
        rng = np.random.RandomState(0)
        series = _get_series(context_df, meta).to_numpy()
        out = []
        cur = series[-1]
        for _ in range(horizon):
            cur = cur + float(rng.normal(0, np.std(np.diff(series)) + 1e-6))
            out.append(cur)
        return PredictionResult(point=out)


# --------------------------------------------------------------------------- #
# Statistical
# --------------------------------------------------------------------------- #
class ARIMAModel(BaseModel):
    """ARIMA via pmdarima. Requires the `pmdarima` extra."""

    def __init__(self, name="arima", order=(2, 1, 1)):
        super().__init__(name, {"forecast"})
        self.order = order

    def predict(self, context_df, meta, horizon=None):
        horizon = horizon or meta.horizon
        try:
            import pmdarima as pm
        except Exception as e:
            raise RuntimeError(
                f"ARIMAModel requires pmdarima: {e}. Install via: uv pip install pmdarima"
            ) from e
        series = _get_series(context_df, meta).to_numpy()
        model = pm.ARIMA(order=self.order, seasonal=False)
        model.fit(series)
        fc = model.predict(n_periods=horizon)
        return PredictionResult(point=[float(x) for x in np.asarray(fc).ravel()])


# --------------------------------------------------------------------------- #
# TSFMs (graceful degradation)
# --------------------------------------------------------------------------- #
class ChronosModel(BaseModel):
    """Amazon Chronos foundation model for forecasting.

    Uses the chronos-forecasting package. Newer versions (>=2.x) expose the
    Chronos2 pipeline with a (n_series, n_variates, history_length) input
    contract; older versions expose ChronosBoltPipeline / ChronosModel.
    We try them in order and degrade gracefully.
    """

    def __init__(self, name="chronos", context_length: int = 512):
        super().__init__(name, {"forecast"})
        self.context_length = context_length

    def _load(self):
        # Prefer the current chronos 2.x pipeline; fall back to bolt, then legacy.
        try:
            from chronos import Chronos2Pipeline  # type: ignore
            return "chronos2", Chronos2Pipeline.from_pretrained(_local_weights("amazon/chronos-2"))
        except Exception:
            pass
        try:
            from chronos import ChronosBoltPipeline  # type: ignore
            return "bolt", ChronosBoltPipeline.from_pretrained(_local_weights("amazon/chronos-bolt-base"))
        except Exception:
            pass
        # legacy
        from chronos import ChronosModel as Legacy  # type: ignore
        return "legacy", Legacy.from_pretrained(_local_weights("amazon/chronos-t5-small"))

    def predict(self, context_df, meta, horizon=None):
        horizon = horizon or meta.horizon
        try:
            kind, model = self._load()
        except Exception as e:
            raise RuntimeError(
                f"ChronosModel requires chronos-forecasting: {e}. "
                "Install via: uv pip install chronos-forecasting"
            ) from e

        series = _get_series(context_df, meta).to_numpy()[-self.context_length:]
        horizon = min(horizon, 200)  # keep reasonable

        if kind == "chronos2":
            # 3D: (n_series, n_variates, history_length)
            inp = np.asarray(series).reshape(1, 1, len(series))
            fc = model.predict(inp, prediction_length=horizon)
            arr = np.asarray(fc)
            # arr shape ~ (n_series, n_samples, horizon); take median over samples
            arr = np.squeeze(arr)
            if arr.ndim == 2:
                fc_med = np.median(arr, axis=0)
            else:
                fc_med = arr
            return PredictionResult(point=[float(x) for x in np.asarray(fc_med).ravel()])

        if kind == "bolt":
            # BoltPipeline.predict(self, inputs, prediction_length=None)
            inp = np.asarray(series)
            fc = model.predict(inp, prediction_length=horizon)
            arr = np.asarray(fc).ravel()
            return PredictionResult(point=[float(x) for x in arr[:horizon]])

        # legacy pipeline
        samples = model.predict(np.asarray(series), prediction_length=horizon,
                                num_samples=20)
        median = np.median(np.asarray(samples), axis=0)
        return PredictionResult(point=[float(x) for x in median])


class TimesFMModel(BaseModel):
    """Google TimesFM foundation model (pip `timesfm` 2.0.x == TimesFM 2.5-200M, PyTorch).

    Fixes vs the original stub: realistic max_context (512, the official GIFT-Eval
    setting), max_horizon 128 (the model's output-patch size), real quantile output
    (levels 0.1..0.9 -> CRPS / VaR / ES), window batching, and a per-process model
    cache so a benchmark run loads the 200M weights once.
    """

    _CACHE: dict = {}          # (ckpt, max_context, max_horizon, batch) -> compiled model
    # Channel layout of the 10 output channels, verified empirically by predictive
    # coverage on sp500_daily (see reports/02_timesfm_benchmark.md): channel 0 is NOT a
    # quantile level (it is the flip-invariance reference kept fixed by
    # `flip_quantile_fn` in timesfm_2p5_torch.compile), channels 1..9 = levels 0.1..0.9,
    # and channel 5 (decode_index) is the median == the point forecast.
    POINT_INDEX = 5
    QUANT_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    QUANT_CHANNELS = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def __init__(self, name: str = "timesfm", checkpoint: str | None = None,
                 max_context: int = 512, max_horizon: int = 128,
                 per_core_batch_size: int = 16, normalize_inputs: bool = True):
        super().__init__(name, {"forecast", "trading"})
        self.checkpoint = checkpoint
        self.max_context = int(max_context)
        self.max_horizon = int(max_horizon)
        self.per_core_batch_size = int(per_core_batch_size)
        self.normalize_inputs = bool(normalize_inputs)

    # ---- loading ----------------------------------------------------------- #
    def _cache_key(self):
        return (self.checkpoint, self.max_context, self.max_horizon,
                self.per_core_batch_size, self.normalize_inputs)

    @classmethod
    def clear_cache(cls) -> None:
        """Release every cached TimesFM instance (each one is ~1 GB of VRAM)."""
        import gc
        try:
            import torch
        except Exception:
            torch = None
        TimesFMModel._CACHE.clear()
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load(self):
        key = self._cache_key()
        if key in TimesFMModel._CACHE:
            return TimesFMModel._CACHE[key]
        # 6 GB GPU: keep at most one compiled instance resident, then evict LRU.
        while len(TimesFMModel._CACHE) >= 1:
            old_key = next(iter(TimesFMModel._CACHE))
            print(f"[timesfm] evicting cached model {old_key} to free VRAM", flush=True)
            del TimesFMModel._CACHE[old_key]
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        import timesfm  # type: ignore
        if self.checkpoint:
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                self.checkpoint, torch_compile=False)
        else:
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                _local_weights(timesfm.TimesFM_2p5_200M_torch.DEFAULT_REPO_ID),
                torch_compile=False)
        model.compile(timesfm.ForecastConfig(
            max_context=self.max_context, max_horizon=self.max_horizon,
            normalize_inputs=self.normalize_inputs, infer_is_positive=False,
            force_flip_invariance=True, per_core_batch_size=self.per_core_batch_size,
            use_continuous_quantile_head=False, fix_quantile_crossing=True))
        TimesFMModel._CACHE[key] = model
        return model

    # ---- forecasting ------------------------------------------------------- #
    def forecast_batch(self, series_list: list[np.ndarray], horizon: int):
        """Batched forecast. Returns (points (N,h), quantiles (N,h,9 levels))."""
        horizon = int(horizon)
        if horizon > self.max_horizon:
            raise ValueError(f"horizon {horizon} > max_horizon {self.max_horizon}")
        model = self._load()
        pts, qs = model.forecast(horizon=horizon, inputs=[np.asarray(x, dtype=np.float64)
                                                          for x in series_list])
        pts = np.asarray(pts, dtype=float)[:, :horizon]
        Q = np.asarray(qs, dtype=float)[:, :horizon, :]
        q9 = Q[:, :, self.QUANT_CHANNELS] if Q.shape[-1] > 5 else Q
        return pts, q9

    def predict(self, context_df, meta, horizon=None):
        horizon = int(horizon or meta.horizon)
        series = _get_series(context_df, meta).to_numpy()
        pts, q9 = self.forecast_batch([series], horizon)
        return PredictionResult(
            point=[float(x) for x in pts[0]],
            quantiles=[[float(x) for x in row] for row in q9[0].T])   # (9, horizon)

    # ---- embeddings (for linear probes) ------------------------------------ #
    def encode(self, series: np.ndarray, max_patches: int = 8) -> np.ndarray:
        """Mean-pooled decoder embeddings of the last `max_patches` patches.

        Replicates the prefill normalization of TimesFM_2p5_200M_torch_module.decode
        so embeddings live in the same space the model was trained in.
        """
        import torch
        from timesfm.torch import util as tfutil
        from timesfm import timesfm_2p5  # noqa: F401

        wrapper = self._load()
        module = wrapper.model
        x = np.asarray(series, dtype=np.float64)
        p = module.p
        keep = min(len(x), p * max(self.max_context // p, max_patches))
        x = x[-keep:]
        pad = (-x.size) % p
        if pad:
            x = np.concatenate([np.full(pad, x[0]), x])
        dev = next(module.parameters()).device
        with torch.no_grad():
            raw = torch.tensor(x, dtype=torch.float32, device=dev).unsqueeze(0)
            mu0 = raw.mean(dim=-1, keepdim=True)
            sd0 = raw.std(dim=-1, keepdim=True)
            if self.normalize_inputs:
                raw = tfutil.revin(raw, mu0, sd0, reverse=False)
            patched = torch.reshape(raw, (1, -1, p))
            masks = torch.zeros_like(patched, dtype=torch.bool)
            n = torch.zeros(1, device=dev)
            mu = torch.zeros(1, device=dev)
            sg = torch.zeros(1, device=dev)
            mus, sgs = [], []
            for i in range(patched.shape[1]):
                (n, mu, sg), _ = tfutil.update_running_stats(n, mu, sg, patched[:, i], masks[:, i])
                mus.append(mu); sgs.append(sg)
            cmu = torch.stack(mus, dim=1); csd = torch.stack(sgs, dim=1)
            normed = tfutil.revin(patched, cmu, csd, reverse=False)
            emb, _ = module(normed, masks, None)
            out = emb[1]                       # output_embeddings (1, P, d_model)
            return out[0, -max_patches:, :].float().cpu().numpy().mean(axis=0)


class TimesFMFinetunedModel(TimesFMModel):
    """TimesFM fine-tuned on in-domain data (checkpoint on disk), same inference path."""

    def __init__(self, name: str = "timesfm_finetuned", checkpoint: str = "", **kw):
        if not checkpoint:
            raise ValueError("TimesFMFinetunedModel requires a checkpoint path")
        super().__init__(name=name, checkpoint=checkpoint, **kw)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def _build(name: str) -> BaseModel:
    name = (name or "").lower().strip()
    if name in ("moving_average", "movingaverage", "ma"):
        return MovingAverageModel()
    if name in ("seasonal_naive", "seasonalnaive", "snaive"):
        return SeasonalNaiveModel()
    if name in ("naive", "naivemodel"):
        return NaiveModel()
    if name in ("random_walk", "random", "randomwalk"):
        return RandomWalkModel()
    if name in ("arima", "arima_model"):
        return ARIMAModel()
    if name in ("chronos", "chronos-t5", "amazon_chronos"):
        return ChronosModel()
    if name in ("timesfm", "timesfm-2.0", "google_timesfm", "timesfm-2.5"):
        return TimesFMModel()
    if name.startswith("timesfm_ft:") or name in ("timesfm_finetuned", "timesfm-ft"):
        ck = name.split(":", 1)[1] if ":" in name else ""
        return TimesFMFinetunedModel(checkpoint=ck)
    raise ValueError(f"Unknown model adapter: {name!r}. "
                     f"Known: {list_available()}")


def list_available() -> list[str]:
    return ["moving_average", "seasonal_naive", "naive", "random_walk",
            "arima", "chronos", "timesfm", "timesfm_finetuned"]


# Default benchmark set (subset that works out-of-the-box; TSFMs need extra deps)
DEFAULT_BENCHMARKS = [
    "moving_average", "seasonal_naive", "naive", "random_walk", "arima",
]
OPTIONAL_BENCHMARKS = ["chronos", "timesfm"]


def available_models() -> dict[str, dict]:
    """Return metadata for each model (including whether deps are installed)."""
    from importlib.util import find_spec
    info = {}
    for name in list_available():
        dep = {
            "arima": "pmdarima",
            "chronos": "chronos",
            "timesfm": "timesfm",
            "timesfm_finetuned": "timesfm",
        }.get(name, None)
        installed = True
        if dep:
            installed = find_spec(dep) is not None
        info[name] = {"name": name, "installed": installed,
                      "requires": dep,
                      "benchmark": name in DEFAULT_BENCHMARKS}
    return info