"""Built-in tiny data so the library (and its tests) can prove a fine-tune works
with nothing downloaded.

``tiny_series`` is a synthetic mean-reverting-with-trend signal; ``tiny_finance``
adds changing volatility and a jump (so the extreme/direction labels are not
degenerate); ``tiny_panel`` is the OHLCV frame the K-line models need. Real datasets
are of course preferred for measurement — these exist for *wiring* checks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .series import SeriesDataset


def tiny_series(n: int = 600, seed: int = 0, name: str = "tiny") -> SeriesDataset:
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    x = (100.0 + 0.05 * t + 6.0 * np.sin(2 * np.pi * t / 50.0)
         + rng.normal(0, 1.2, n)).cumsum() * 0.02 + 100.0 + 0.08 * t
    return _wrap(name, np.asarray(x, dtype=float), n)


def tiny_finance(n: int = 900, seed: int = 1, name: str = "tiny_fin") -> SeriesDataset:
    """Regime-switching log-price: low-vol → high-vol → jump → calm."""
    rng = np.random.RandomState(seed)
    lr = np.zeros(n)
    vol = np.r_[np.full(n // 3, 0.006), np.full(n // 3, 0.02), np.full(n - 2 * (n // 3), 0.01)]
    lr += rng.normal(0.0003, vol)
    k = int(0.7 * n)
    lr[k] -= 0.06                                   # a crash day
    price = 50.0 * np.exp(np.cumsum(lr))
    df = pd.DataFrame({"timestamp": pd.date_range("2015-01-01", periods=n, freq="B"),
                       "close": price})
    df["open"] = df["close"].shift(1).fillna(df["close"].iloc[0])
    df["high"] = np.maximum(df["open"], df["close"]) * (1 + np.abs(rng.normal(0, 0.004, n)))
    df["low"] = np.minimum(df["open"], df["close"]) * (1 - np.abs(rng.normal(0, 0.004, n)))
    df["volume"] = np.abs(rng.normal(1e6, 2e5, n))
    a, b = int(0.7 * n), int(0.85 * n)
    return SeriesDataset(name=name, target="close", values=df["close"].to_numpy(float),
                         index=pd.DatetimeIndex(df["timestamp"]),
                         panel=df.drop(columns=["close"]),
                         splits={"train": (0, a), "validation": (a, b), "test": (b, n)},
                         meta={"n_obs": n, "frequency": "day", "source": "fixture",
                               "panel_cols": ["open", "high", "low", "volume"]})


def _wrap(name, x, n):
    a, b = int(0.7 * n), int(0.85 * n)
    return SeriesDataset(name=name, target="value", values=np.asarray(x, float),
                         splits={"train": (0, a), "validation": (a, b), "test": (b, n)},
                         meta={"n_obs": n, "source": "fixture"})


def tiny_windows(n: int = 8, ctx: int = 128, horizon: int = 16, seed: int = 0):
    from .series import windows
    ds = tiny_series(seed=seed)
    w = windows(ds.split("train"), horizon, ctx, stride=max(1, ctx // 2))
    return w[:n] if len(w) >= n else w
