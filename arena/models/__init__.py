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

from ..model_base import BaseModel, PredictionResult
from ..schemas import DatasetMeta, TaskType


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
            return "chronos2", Chronos2Pipeline.from_pretrained("amazon/chronos-2")
        except Exception:
            pass
        try:
            from chronos import ChronosBoltPipeline  # type: ignore
            return "bolt", ChronosBoltPipeline.from_pretrained("amazon/chronos-bolt-base")
        except Exception:
            pass
        # legacy
        from chronos import ChronosModel as Legacy  # type: ignore
        return "legacy", Legacy.from_pretrained("amazon/chronos-t5-small")

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
    """Google TimesFM foundation model (pip: timesfm)."""

    def __init__(self, name="timesfm"):
        super().__init__(name, {"forecast"})

    def predict(self, context_df, meta, horizon=None):
        horizon = horizon or meta.horizon
        try:
            from timesfm import TimeSeriesModel  # type: ignore
        except Exception as e:
            raise RuntimeError(
                f"TimesFMModel requires timesfm: {e}. "
                "Install via: uv pip install timesfm"
            ) from e
        series = _get_series(context_df, meta).to_numpy()
        model = TimeSeriesModel.from_pretrained("google/timeseries-tfm-2.0")
        samples = model.forecast(np.array([series]))
        median = np.median(np.asarray(samples), axis=0).ravel()
        out = median[:horizon].tolist()
        return PredictionResult(point=out)


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
    if name in ("timesfm", "timesfm-2.0", "google_timesfm"):
        return TimesFMModel()
    raise ValueError(f"Unknown model adapter: {name!r}. "
                     f"Known: {list_available()}")


def list_available() -> list[str]:
    return ["moving_average", "seasonal_naive", "naive", "random_walk",
            "arima", "chronos", "timesfm"]


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
        }.get(name, None)
        installed = True
        if dep:
            installed = find_spec(dep) is not None
        info[name] = {"name": name, "installed": installed,
                      "requires": dep,
                      "benchmark": name in DEFAULT_BENCHMARKS}
    return info