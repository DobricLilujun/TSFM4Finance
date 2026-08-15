"""Dataset builder for Finance TSFMs Arena.

Generates the open and closed datasets that the arena evaluates on.

Open datasets: real, freely-available financial time series
  - equity   : S&P 500 daily (via yfinance) + a synthetic stock (fallback)
  - crypto   : BTC/ETH/... via yfinance (fallback synthetic)
  - fx       : EUR/USD via yfinance (fallback synthetic)
  - index    : S&P 500 index
  - anomaly  : a synthetic "fraud/transaction" stream with injected anomalies
  - classify : a synthetic series whose sign of returns is a binary label

Closed datasets: placeholders that mirror the SAME schema but contain NO real
  rows beyond a tiny synthetic stub — marked requires_download=True so the
  arena knows the user must supply the data (or upload their own model).

Every dataset is written as:
    <root>/<open|closed>/<name>/meta.json
    <root>/<open|closed>/<name>/train.parquet
    <root>/<open|closed>/<name>/validation.parquet
    <root>/<open|closed>/<name>/test.parquet

Splitting is strictly chronological (no leakage): train | validation | test.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from arena.schemas import DatasetMeta, Domain, TaskType, Frequency

ROOT = Path(__file__).resolve().parent.parent / "data"


# --------------------------------------------------------------------------- #
# Generators
# --------------------------------------------------------------------------- #
def _geometric_series(n: int, start: float, mu: float, sigma: float,
                      seed: int, freq: str) -> pd.Series:
    """Geometric (lognormal) return path — realistic for prices."""
    rng = np.random.RandomState(seed)
    returns = rng.normal(mu, sigma, n)
    price = start * np.cumprod(1 + returns)
    idx = pd.date_range("2019-01-01", periods=n, freq=freq)
    s = pd.Series(price, index=idx, name="close")
    return s


def _add_seasonality(series: pd.Series, amp: float, period: int) -> pd.Series:
    n = len(series)
    seas = amp * np.sin(np.arange(n) / period * 2 * np.pi)
    return series * (1 + seas / 100.0)


def _inject_anomalies(series: pd.Series, n_anom: int,
                      seed: int = 99) -> pd.Series:
    rng = np.random.RandomState(seed)
    s = series.copy()
    positions = rng.choice(len(s), size=n_anom, replace=False)
    s.iloc[positions] *= 2.5 + rng.rand(n_anom)
    return s


def _classify_series(n: int, seed: int, freq: str) -> pd.DataFrame:
    """Series with a binary 'label' = sign of next-day return."""
    rng = np.random.RandomState(seed)
    ret = rng.normal(0.0003, 0.02, n)
    price = 100 * np.cumprod(1 + ret)
    label = (ret > 0).astype(int)
    idx = pd.date_range("2018-01-01", periods=n, freq=freq)
    df = pd.DataFrame({"timestamp": idx, "close": price, "label": label})
    return df


def _anomaly_series(n: int, seed: int, freq: str) -> pd.DataFrame:
    """Transaction/flow stream with injected anomalies (value = 1 if anom)."""
    rng = np.random.RandomState(seed)
    base = np.abs(rng.normal(0, 1, n)) + 1.0
    anomaly = np.zeros(n)
    positions = rng.choice(n, size=int(n * 0.03), replace=False)
    anomaly[positions] = 1.0
    # anomalies show as large spikes
    flow = base * (1 + anomaly * 8)
    idx = pd.date_range("2022-01-01", periods=n, freq=freq)
    df = pd.DataFrame({"timestamp": idx, "flow": flow, "label": anomaly})
    return df


# --------------------------------------------------------------------------- #
# Download (open, real) with synthetic fallback
# --------------------------------------------------------------------------- #
def _download_yfinance(ticker: str, period: str = "2y") -> pd.DataFrame | None:
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df is None or len(df) < 150:
            return None
        df = df.reset_index()
        # yfinance may return a MultiIndex column; flatten it.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if len(c) == 1 else c[1] for c in df.columns.values]
        col = "Close" if "Close" in df.columns else None
        if col is None:
            # pick the first price-like column
            price_cols = [c for c in df.columns if any(k in str(c).lower()
                             for k in ("close", "adj", "price"))]
            col = price_cols[0] if price_cols else df.columns[-1]
        out = pd.DataFrame({"timestamp": df["Date"], "close": df[col]})
        out = out.dropna().reset_index(drop=True)
        return out if len(out) >= 150 else None
    except Exception:
        return None


def _to_dataframe(series: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"timestamp": series.index, "close": series.values})


# --------------------------------------------------------------------------- #
# Splits (strict chronological)
# --------------------------------------------------------------------------- #
def _split(df: pd.DataFrame, target: str,
           train_frac: float = 0.7, val_frac: float = 0.15,
           lookback: int = 120, horizon: int = 10):
    """Split into train/val/test with a lookback overlap on train for context."""
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    # train keeps a lookback window at its tail so the context contract holds
    train = df.iloc[:train_end + lookback].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test


# --------------------------------------------------------------------------- #
# Dataset definitions
# --------------------------------------------------------------------------- #
OPEN_DEFS = [
    dict(name="sp500_daily", domain=Domain.INDEX, task=TaskType.FORECAST,
         ticker="^GSPC", freq=Frequency.DAY, lookback=120, horizon=10,
         source="S&P 500 via yfinance", seed=1, synthetic_mu=0.0003,
         synthetic_sigma=0.012),
    dict(name="equity_synthetic", domain=Domain.EQUITY, task=TaskType.FORECAST,
         ticker=None, freq=Frequency.DAY, lookback=120, horizon=5,
         source="Synthetic stock (fallback)", seed=2, synthetic_mu=0.0004,
         synthetic_sigma=0.015),
    dict(name="btc_crypto", domain=Domain.CRYPTO, task=TaskType.FORECAST,
         ticker="BTC-USD", freq=Frequency.DAY, lookback=150, horizon=7,
         source="BTC-USD via yfinance", seed=3, synthetic_mu=0.0006,
         synthetic_sigma=0.045),
    dict(name="eur_usd_fx", domain=Domain.FX, task=TaskType.FORECAST,
         ticker="EURUSD=X", freq=Frequency.DAY, lookback=120, horizon=5,
         source="EUR/USD via yfinance", seed=4, synthetic_mu=0.0,
         synthetic_sigma=0.006),
    dict(name="fraud_anomaly", domain=Domain.FRAUD, task=TaskType.ANOMALY,
         ticker=None, freq=Frequency.HOUR, lookback=100, horizon=24,
         source="Synthetic transaction/flow stream", seed=5),
    dict(name="return_classify", domain=Domain.EQUITY, task=TaskType.CLASSIFY,
         ticker=None, freq=Frequency.DAY, lookback=60, horizon=1,
         source="Synthetic return sign classifier", seed=6),
]

# Closed datasets: same schema, placeholder (no real rows) — user must provide.
CLOSED_DEFS = [
    dict(name="priv_fund_flow", domain=Domain.EQUITY, task=TaskType.FORECAST,
         freq=Frequency.DAY, lookback=120, horizon=5,
         source="PROPRIETARY — requires user data / model upload", seed=11),
    dict(name="priv_broker_order", domain=Domain.EQUITY, task=TaskType.ANOMALY,
         freq=Frequency.MINUTE, lookback=60, horizon=10,
         source="PROPRIETARY — requires user data / model upload", seed=12),
    dict(name="priv_credit_risk", domain=Domain.BOND, task=TaskType.CLASSIFY,
         freq=Frequency.DAY, lookback=60, horizon=1,
         source="PROPRIETARY — requires user data / model upload", seed=13),
    dict(name="priv_crypto_tick", domain=Domain.CRYPTO, task=TaskType.FORECAST,
         freq=Frequency.MINUTE, lookback=60, horizon=10,
         source="PROPRIETARY — requires user data / model upload", seed=14),
]


# --------------------------------------------------------------------------- #
def _build_open(defn: dict) -> DatasetMeta:
    freq_map = {Frequency.MINUTE: "1min", Frequency.HOUR: "1h",
                Frequency.DAY: "1D", Frequency.WEEK: "1W"}
    f = freq_map[defn["freq"]]

    # 1. try to download real open data
    df = None
    if defn.get("ticker"):
        df = _download_yfinance(defn["ticker"], period="3y")

    if df is None:
        # synthetic fallback
        if defn["task"] == TaskType.ANOMALY:
            df = _anomaly_series(n=1000, seed=defn["seed"], freq=f)
            target = "flow"
            labels = ["flow", "label"]
        elif defn["task"] == TaskType.CLASSIFY:
            df = _classify_series(n=1000, seed=defn["seed"], freq=f)
            target = "label"
            labels = ["close", "label"]
        else:  # forecast
            s = _geometric_series(n=1000, start=100,
                                  mu=defn["synthetic_mu"],
                                  sigma=defn["synthetic_sigma"],
                                  seed=defn["seed"], freq=f)
            s = _add_seasonality(s, amp=defn["synthetic_sigma"] * 5, period=20)
            df = _to_dataframe(s)
            target = "close"
            labels = ["close"]
        source = defn["source"] + " (synthetic fallback)"
    else:
        target = "close"
        labels = ["close"]
        source = defn["source"]

    # 2. chronological split
    train, val, test = _split(df, target, lookback=defn["lookback"],
                               horizon=defn["horizon"])

    n_obs = len(df)
    meta = DatasetMeta(
        name=defn["name"], domain=defn["domain"], task=defn["task"],
        frequency=defn["freq"], n_features=1, feature_names=labels,
        n_obs=n_obs, target=target, lookback=defn["lookback"],
        horizon=defn["horizon"], open=True, requires_download=False,
        source=source, license="open",
        notes="Open dataset. Real data when available; synthetic fallback otherwise.",
    )
    _write(defn["name"], meta, train, val, test, closed=False)
    return meta


def _build_closed(defn: dict) -> DatasetMeta:
    freq_map = {Frequency.MINUTE: "1min", Frequency.HOUR: "1h",
                Frequency.DAY: "1D", Frequency.WEEK: "1W"}
    f = freq_map[defn["freq"]]

    # larger synthetic stub so the evaluate/submit path produces a
    # non-degenerate score. STILL a placeholder: no real proprietary data.
    n = 600
    if defn["task"] == TaskType.ANOMALY:
        df = _anomaly_series(n=n, seed=defn["seed"], freq=f)
        target = "flow"
    elif defn["task"] == TaskType.CLASSIFY:
        df = _classify_series(n=n, seed=defn["seed"], freq=f)
        target = "label"
    else:
        s = _geometric_series(n=n, start=100, mu=0.0003, sigma=0.012,
                              seed=defn["seed"], freq=f)
        s = _add_seasonality(s, amp=0.012 * 5, period=20)
        df = _to_dataframe(s)
        target = "close"

    # chronological split (same helper as open)
    train, val, test = _split(df, target, lookback=defn["lookback"],
                               horizon=defn["horizon"])

    meta = DatasetMeta(
        name=defn["name"], domain=defn["domain"], task=defn["task"],
        frequency=defn["freq"], n_features=1,
        feature_names=list(df.columns[1:]),
        n_obs=len(df), target=target, lookback=defn["lookback"],
        horizon=defn["horizon"], open=False, requires_download=True,
        source=defn["source"], license="proprietary",
        notes="CLOSED/proprietary placeholder. Contains NO real data. "
              "The user must provide the data or upload their own model to be "
              "evaluated; until then only the synthetic stub is present.",
    )
    _write(defn["name"], meta, train, val, test, closed=True)
    return meta


def _write(name: str, meta: DatasetMeta, train: pd.DataFrame,
           val: pd.DataFrame, test: pd.DataFrame, closed: bool):
    bucket = "closed" if closed else "open"
    d = ROOT / bucket / name
    d.mkdir(parents=True, exist_ok=True)
    train.to_parquet(d / "train.parquet")
    val.to_parquet(d / "validation.parquet")
    test.to_parquet(d / "test.parquet")
    (d / "meta.json").write_text(meta.model_dump_json(indent=2))


# --------------------------------------------------------------------------- #
def build_all() -> list[DatasetMeta]:
    metas = []
    for d in OPEN_DEFS:
        try:
            metas.append(_build_open(d))
        except Exception as e:
            print(f"[warn] open {d['name']} failed: {e}")
    for d in CLOSED_DEFS:
        try:
            metas.append(_build_closed(d))
        except Exception as e:
            print(f"[warn] closed {d['name']} failed: {e}")
    print(f"Built {len(metas)} datasets -> {ROOT}")
    return metas


if __name__ == "__main__":
    build_all()