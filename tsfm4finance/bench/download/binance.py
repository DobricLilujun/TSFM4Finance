"""T3 intraday finance data: Binance bulk monthly klines (no key, no auth).

Daily bars give 5-8 non-overlapping windows per year at h=21; hourly bars give
~170.  For a 15-model sweep that is the difference between a ranking and noise,
and intraday is also where "low signal-to-noise" stops being a rhetorical point.
Files are cached one month at a time, so an interrupted download resumes.
"""
from __future__ import annotations

import io
import time

import pandas as pd
import requests

from .common import RAW, dataset_exists, record_failure, clear_failure, retry, write_open_dataset

BASE = "https://data.binance.vision/data/spot/monthly/klines"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]

PAIRS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
         "DOGEUSDT", "LINKUSDT", "LTCUSDT", "AVAXUSDT", "DOTUSDT", "TRXUSDT"]


def _month_path(symbol: str, interval: str, ym: str):
    return RAW / f"{symbol}_{interval}_{ym}.csv"


@retry(times=3, base_delay=2.0)
def fetch_month(symbol: str, interval: str, ym: str) -> bool:
    """True if the file is present/created, False if that month does not exist."""
    p = _month_path(symbol, interval, ym)
    if p.exists() and p.stat().st_size > 1000:
        return True
    url = f"{BASE}/{symbol}/{interval}/{symbol}-{interval}-{ym}.zip"
    r = requests.get(url, timeout=120)
    if r.status_code == 404 or len(r.content) < 200:
        return False
    df = pd.read_csv(io.BytesIO(r.content), header=None)
    if str(df.iloc[0, 0]).strip().lower().startswith("open"):
        df = df.iloc[1:]
    df.columns = COLS
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return True


def months(start: str, end: str | None = None):
    ts = pd.period_range(start, end or pd.Timestamp.utcnow().strftime("%Y-%m"), freq="M")
    return [str(t) for t in ts]


def build_dataset(symbol: str, interval: str = "1h", start: str = "2019-01",
                  end: str | None = None, *, force: bool = False,
                  sleep: float = 0.05) -> str:
    name = f"bin_{symbol.lower()}_{interval}"
    if dataset_exists(name) and not force:
        return "exists"
    frames, missing = [], []
    for ym in months(start, end):
        try:
            if fetch_month(symbol, interval, ym):
                frames.append(pd.read_csv(_month_path(symbol, interval, ym)))
            else:
                missing.append(ym)
        except Exception as e:                                 # noqa: BLE001
            missing.append(f"{ym}:{e}")
        time.sleep(sleep)
    if not frames:
        record_failure("binance", name, f"no months fetched (missing={missing[:6]})")
        return f"failed:no data; missing={missing[:6]}"
    df = pd.concat(frames, ignore_index=True)
    for c in ("open", "high", "low", "close", "volume", "quote_volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    ot = pd.to_datetime(pd.to_numeric(df["open_time"], errors="coerce"), unit="ms", utc=True)
    if pd.to_numeric(df["open_time"], errors="coerce").median() < 1e11:
        ot = pd.to_datetime(pd.to_numeric(df["open_time"], errors="coerce"), unit="s", utc=True)
    out = pd.DataFrame({"timestamp": ot, "open": df["open"], "high": df["high"],
                        "low": df["low"], "close": df["close"], "volume": df["volume"],
                        "amount": df["quote_volume"]}).dropna(subset=["close"])
    out = out[~out.timestamp.duplicated()].sort_values("timestamp").reset_index(drop=True)
    if len(out) < 2000:
        record_failure("binance", name, f"only {len(out)} bars")
        return f"failed:too few bars ({len(out)})"
    freq = {"1m": "minute", "5m": "5minute", "15m": "15minute", "1h": "hour", "4h": "4hour"}[interval]
    write_open_dataset(
        name, out, freq=freq, domain="crypto", source=f"binance:{symbol}:{interval}",
        license_="exchange market data", notes=f"UTC bars, {start}->present; OHLCV+quote "
        "volume; chronological 70/15/15 splits",
        extra_meta={"ohlcv": True, "interval": interval, "missing_months": missing[:12]},
    )
    clear_failure("binance", name)
    return "ok"


def run(interval: str = "1h", start: str = "2019-01", pairs: list[str] | None = None,
        force: bool = False) -> dict:
    return {f"{s}_{interval}": build_dataset(s, interval, start, force=force)
            for s in (pairs or PAIRS)}
