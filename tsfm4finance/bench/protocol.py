"""Benchmark protocol for the Finance TSFM Arena (spec §4.1).

Protocol decisions (see TSFM_Paper/plans/timesfm-benchmark.md, Q4):

* horizon h is part of the experiment, not a dataset constant: h in {1, 5, 12, 21, 63}
  (1d / 1w / arena-continuity / 1m / 1q of business days).
* rolling-origin backtest with **non-overlapping windows, stride = h** (GIFT-Eval /
  Chronos convention: "all disjoint horizons in the test period").
* evaluation span: the **test** split for short horizons (h <= 22); **val+test** for long
  horizons, because a 113-row test split holds only ~1.8 windows at h=63.
* context: **expanding history capped at max_context = 512** (the official TimesFM
  GIFT-Eval setting), always ending strictly before the origin -> no leakage.
* metrics averaged across origins (mean +/- std); MASE scale from the TRAIN split,
  season = 1 (business-day daily data).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

MAX_CONTEXT = 512
HORIZONS = [1, 5, 12, 21, 63]
LONG_HORIZON_SWITCH = 22      # h > this -> evaluate over val+test
MIN_CONTEXT = 32              # TimesFM input patch size


@dataclass
class Window:
    """One decision point: predict `truth` from `context`.

    ``panel`` / ``truth_panel`` carry the full OHLCV block for the same window.
    They exist because some finance-native models (Kronos) consume candlesticks
    rather than a single close series; models that do not need them simply ignore
    them, and the metric layer always scores the same ``target`` column.
    """
    origin: int            # index in the full (train+val+test) array where truth starts
    context: np.ndarray
    truth: np.ndarray
    panel: object | None = None
    truth_panel: object | None = None


def build_windows(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                  horizon: int, target: str = "close",
                  max_context: int = MAX_CONTEXT,
                  stride: int | None = None,
                  span: str = "auto") -> tuple[list[Window], dict]:
    """Non-overlapping rolling-origin windows over the evaluation span.

    Returns (windows, info) where info records span, n_origins, coverage and an
    explicit no-leakage assertion result.
    """
    full = pd.concat([train, val, test], ignore_index=True)
    y = full[target].to_numpy(dtype=float)
    n_train, n_val, n_test = len(train), len(val), len(test)
    n = len(y)

    panel_cols = [c for c in full.columns if c != target]
    with_panel = bool(panel_cols)

    if span == "auto":
        span = "val+test" if horizon > LONG_HORIZON_SWITCH else "test"
    if span == "test":
        span_start, span_end = n_train + n_val, n
    elif span == "val":
        # strictly the validation split: an untouched span that fine-tuning never saw
        span_start, span_end = n_train, n_train + n_val
    elif span == "val+test":
        span_start, span_end = n_train, n
    elif span == "train+val+test":
        span_start, span_end = 0, n
    else:
        raise ValueError(f"unknown span {span!r}")

    stride = int(stride or horizon)
    want_panel = any(c != target for c in full.columns)      # OHLCV panels for K-line models
    windows: list[Window] = []
    o = span_start
    last_end = span_start
    while o + horizon <= span_end:
        ctx_lo = max(0, o - max_context)
        ctx = y[ctx_lo:o]
        truth = y[o:o + horizon]
        if len(ctx) >= MIN_CONTEXT:
            w = Window(origin=o, context=ctx, truth=truth)
            if want_panel:
                w.panel = full.iloc[ctx_lo:o]
                w.truth_panel = full.iloc[o:o + horizon]
            windows.append(w)
        o += stride
        last_end = o
    # a trailing partial window would break the non-overlap / fixed-h contract, so we
    # simply report how much of the span was covered instead.

    covered = sum(len(w.truth) for w in windows)
    span_len = span_end - span_start
    info = {
        "span": span,
        "horizon": int(horizon),
        "stride": int(stride),
        "n_origins": len(windows),
        "coverage": float(covered / span_len) if span_len else float("nan"),
        "max_context": int(max_context),
        "no_leakage": all((w.origin + len(w.truth) <= span_end) and (len(w.context) == w.origin - max(0, w.origin - max_context))
                          for w in windows),
        "train_end": int(n_train),        # everything at/after this index is unseen by training
    }
    return windows, info


def aggregate(per_window: list[dict], prefix: str = "") -> dict:
    """mean +/- std of every numeric metric across windows."""
    if not per_window:
        return {}
    keys = [k for k in per_window[0] if isinstance(per_window[0][k], (int, float))]
    out: dict = {}
    for k in keys:
        vals = [w[k] for w in per_window]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            out[f"{prefix}{k}"] = float("nan")
            out[f"{prefix}{k}_std"] = float("nan")
            continue
        out[f"{prefix}{k}"] = float(np.mean(vals))
        out[f"{prefix}{k}_std"] = float(np.std(vals))
    out[f"{prefix}n_origins"] = len(per_window)
    return out


# --------------------------------------------------------------------------- #
# Batched model engines
#
# `predict_batch(contexts, h) -> (points (N,h), quantiles (N,h,9) or None)`
# Engines, unlike arena adapters, evaluate a whole window set in one call.
# --------------------------------------------------------------------------- #
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class Engine:
    name = "engine"

    def predict_batch(self, contexts: Sequence[np.ndarray], horizon: int):
        raise NotImplementedError


class NaiveEngine(Engine):
    """Persistence: repeat the last observed value."""
    name = "naive"

    def predict_batch(self, contexts, horizon):
        last = np.array([float(c[-1]) for c in contexts])[:, None]
        pts = np.repeat(last, horizon, axis=1)
        return pts, None


class SeasonalNaiveEngine(Engine):
    """Repeat the last `season` observations (season=5 = last business week)."""
    name = "seasonal_naive"

    def __init__(self, season: int = 5):
        self.season = season

    def predict_batch(self, contexts, horizon):
        pts = []
        for c in contexts:
            c = np.asarray(c, dtype=float)
            s = self.season
            if len(c) < s:
                pts.append([float(c[-1])] * horizon)
                continue
            tail = c[-s:]
            pts.append([float(tail[i % s]) for i in range(horizon)])
        return np.array(pts), None


class MovingAverageEngine(Engine):
    name = "moving_average"

    def __init__(self, window: int = 20):
        self.window = window

    def predict_batch(self, contexts, horizon):
        pts = []
        for c in contexts:
            tail = np.asarray(c, dtype=float)[-self.window:]
            pts.append([float(np.mean(tail))] * horizon)
        return np.array(pts), None


class ARIMAEngine(Engine):
    """ARIMA(p,d,q) via pmdarima, refit at every origin (rolling backtest).

    `refit_every` widens the refit period for very short horizons (many origins) to
    keep the sweep tractable; the model still only ever sees data up to the origin.
    """
    name = "arima"

    def __init__(self, order=(1, 1, 0), refit_every: int | None = None):
        self.order = order
        self.refit_every = refit_every

    def predict_batch(self, contexts, horizon):
        import warnings
        warnings.filterwarnings("ignore")
        import pmdarima as pm
        pts = []
        for k, c in enumerate(contexts):
            c = np.asarray(c, dtype=float)
            # delevel long contexts for numerical stability, then re-level
            base = float(c[-1])
            z = c - base
            model = pm.ARIMA(order=self.order, seasonal=False, suppress_warnings=True,
                             method="lbfgs", maxiter=300)
            try:
                model.fit(z)
                # predictions are in de-leveled space; re-level to price units
                f = np.asarray(model.predict(n_periods=horizon), dtype=float) + base
            except Exception:
                f = np.repeat(float(c[-1]), horizon)
            pts.append(f[:horizon] if len(f) >= horizon else
                       np.r_[f, np.repeat(f[-1] if len(f) else c[-1], horizon - len(f))])
            del model
        return np.array(pts), None


class ChronosEngine(Engine):
    """Amazon Chronos-2 (open TSFM) as the reference foundation-model baseline."""
    name = "chronos"

    def __init__(self, model_id: str = "amazon/chronos-2", max_context: int = MAX_CONTEXT,
                 batch_size: int = 32):
        self.model_id = model_id
        self.max_context = max_context
        self.batch_size = batch_size
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            from chronos import Chronos2Pipeline
            self._pipe = Chronos2Pipeline.from_pretrained(self.model_id, device_map="auto")
        return self._pipe

    def predict_batch(self, contexts, horizon):
        pipe = self._load()
        cs = [np.asarray(c, dtype=np.float32)[-self.max_context:] for c in contexts]
        inp = np.stack([c.reshape(1, -1) for c in cs])          # (N, 1, L) if equal length
        lens = {c.shape[-1] for c in cs}
        if len(lens) != 1:
            inp = [c.reshape(1, -1) for c in cs]
        # chronos-2 returns (quantiles, means); quantiles (N, variates, h, K)
        try:
            q_raw, m_raw = pipe.predict_quantiles(inp, prediction_length=horizon,
                                                  quantile_levels=LEVELS,
                                                  batch_size=self.batch_size)
        except TypeError:      # older signature without batch_size
            q_raw, m_raw = pipe.predict_quantiles(inp, prediction_length=horizon,
                                                  quantile_levels=LEVELS)
        Q = np.asarray(q_raw, dtype=float)
        while Q.ndim > 3:                    # drop variate axes
            Q = Q[:, 0]
        if Q.ndim == 2:
            Q = Q[None, ...]
        pts = np.asarray(m_raw, dtype=float)
        while pts.ndim > 2:
            pts = pts[:, 0]
        if pts.ndim == 1:
            pts = pts[None, :]
        return pts[:, :horizon], Q[:, :horizon, :len(LEVELS)]


class TimesFMEngine(Engine):
    """Google TimesFM 2.5-200M (zero-shot or fine-tuned checkpoint)."""
    name = "timesfm"

    def __init__(self, checkpoint: str | None = None, max_context: int = MAX_CONTEXT,
                 max_horizon: int = 128, per_core_batch_size: int = 16, name: str | None = None,
                 normalize_inputs: bool = True):
        from tsfm4finance.core.models import TimesFMModel, TimesFMFinetunedModel
        if checkpoint:
            self.model = TimesFMFinetunedModel(checkpoint=checkpoint, max_context=max_context,
                                               max_horizon=max_horizon,
                                               per_core_batch_size=per_core_batch_size,
                                               normalize_inputs=normalize_inputs)
            self.name = name or f"timesfm_ft"
        else:
            self.model = TimesFMModel(max_context=max_context, max_horizon=max_horizon,
                                      per_core_batch_size=per_core_batch_size,
                                      normalize_inputs=normalize_inputs)
            self.name = name or "timesfm"
        self.max_context = max_context

    def predict_batch(self, contexts, horizon):
        cs = [np.asarray(c, dtype=float)[-self.max_context:] for c in contexts]
        return self.model.forecast_batch(cs, horizon)

    def encode(self, series, max_patches: int = 8):
        return self.model.encode(series, max_patches=max_patches)
