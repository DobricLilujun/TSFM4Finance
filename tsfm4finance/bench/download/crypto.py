"""Crypto acquisition layer for the Arena (Binance, no API key).

Why crypto is the first thing to expand: §5.3.4 of the survey discusses crypto markets at
length while the arena had exactly two Yahoo series.  And crypto is where the Arena can be
*contamination-controlled*, which a public benchmark normally cannot be:

* ``post_listing`` - the instrument was listed after the plausible pre-training cut-off, so
  its whole history post-dates the corpora a zero-shot TSFM was trained on;
* ``fresh``        - bars from the last weeks (2026-08 -> now), which no released model can
  have seen;
*  ``rare``         - derivatives plumbing (funding rates, mark-price basis) that does not
   appear in public time-series corpora the way spot prices do.

Everything is resumable: raw vendor files land in ``data/raw/`` and are reused, and each
finished dataset is registered in ``data/manifest.json`` with a checksum.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pandas as pd

from .common import (RAW, dataset_exists, manifest_update, record_failure,
                     clear_failure, retry, http_get, write_open_dataset)

VISION = "https://data.binance.vision/data"
SPOT_API = "https://api.binance.com/api/v3"
FUT_API = "https://fapi.binance.com/fapi/v1"

# liquid, long-lived USDT pairs (the "majors" panel)
MAJORS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
          "LINKUSDT", "LTCUSDT", "AVAXUSDT", "DOTUSDT", "TRXUSDT", "SHIBUSDT", "NEARUSDT",
          "APTUSDT", "ARBUSDT", "OPUSDT", "ATOMUSDT", "FILUSDT", "ETCUSDT", "BCHUSDT",
          "XLMUSDT", "ICPUSDT", "UNIUSDT", "AAVEUSDT", "ALGOUSDT", "VETUSDT", "INJUSDT",
          "SUIUSDT", "SEIUSDT", "TIAUSDT", "ENAUSDT", "RNDRUSDT", "FTMUSDT", "HBARUSDT",
          "GRTUSDT", "CRVUSDT", "MINAUSDT", "SANDUSDT", "MANAUSDT"]
# stablecoin peg series: externally documented depeg events = anomaly truth we did NOT invent
PEGS = ["USDCUSDT", "BUSDUSDT", "DAIUSDT", "FDUSDUSDT", "TUSDUSDT", "USDPUSDT"]
CUTOFF = "2025-01-01"          # anything listed after this is clean by construction

KCOLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
         "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]


# ---------------------------------------------------------------- download ---
@retry(times=3, base_delay=2.0)
def _get(url: str, timeout: int = 120) -> bytes:
    return http_get(url, timeout=timeout)


def _ts_col(open_time: pd.Series) -> pd.Series:
    """Normalise Binance's open_time to UTC timestamps.

    Binance changed kline files from millisecond to **second** epochs for 2025+ files, so
    the unit is decided per file by magnitude (s < 1e11 <= ms < 1e14 <= us).
    """
    mx = pd.to_numeric(open_time, errors="coerce").max()
    unit = "s" if mx < 1e11 else ("ms" if mx < 1e14 else "us")
    return pd.to_datetime(pd.to_numeric(open_time, errors="coerce"), unit=unit, utc=True)


def _month_zip_to_df(url: str, cache: str) -> pd.DataFrame | None:
    p = RAW / cache
    if p.exists() and p.stat().st_size > 200:
        raw = p.read_bytes()
    else:
        try:
            raw = _get(url)
        except Exception:                                            # noqa: BLE001
            return None
        if not raw or raw[:5] not in (b"PK\x03\x04", b"time", b"open") and len(raw) < 200:
            return None
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
    if not raw or len(raw) < 200:
        return None
    try:
        if raw[:4] == b"PK\x03\x04":
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                csv = z.read(z.namelist()[0])
        else:
            csv = raw
    except Exception:                                                # noqa: BLE001
        return None
    head = csv[:200].decode("utf-8", "ignore").lstrip()
    if head.startswith(("<", "NoSuchKey", "SignatureDoesNotMatch", '{"code"')):
        return None                                   # 404 / error page, not a bar file
    try:
        df = pd.read_csv(io.BytesIO(csv), header=None)
    except Exception:                                  # noqa: BLE001
        return None
    if df.shape[1] < 8:
        return None
    if str(df.iloc[0, 0]).lower().startswith(("open_time", "open")):
        df = df.iloc[1:]
    df.columns = KCOLS[: df.shape[1]]
    if pd.to_numeric(df["open_time"], errors="coerce").isna().all():
        return None
    return df


def _kline_frame(pair: str, interval: str, start: str, market: str = "spot") -> pd.DataFrame:
    """Concatenate every monthly bar file from ``start`` to now (404 month = skip)."""
    end = pd.Timestamp.now(tz="UTC").strftime("%Y-%m")
    frames, seen = [], set()
    for ym in [str(p) for p in pd.period_range(start, end, freq="M")]:
        sub = "monthly/klines"
        url = f"{VISION}/{market}/{sub}/{pair}/{interval}/{pair}-{interval}-{ym}.zip"
        cache = f"binance/{market}/{pair}/{interval}/{pair}-{interval}-{ym}.zip"
        df = _month_zip_to_df(url, cache)
        if df is None:
            continue
        # unit per FILE: 2024 and earlier are ms, 2025+ are seconds - deciding the unit
        # after concatenation would shift the old rows into 1970
        df["timestamp"] = _ts_col(df["open_time"])
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    for c in ("open", "high", "low", "close", "volume", "quote_volume",
              "trades", "taker_buy_base"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")].reset_index()
    return df[["timestamp", "open", "high", "low", "close", "volume",
               "quote_volume", "trades", "taker_buy_base"]]


def universe(top: int = 400) -> list[str]:
    """USDT pairs ranked by 24 h quote volume - the candidate universe for N5."""
    rows = json.loads(_get(f"{SPOT_API}/ticker/24hr", 90))
    liq = [(float(r["quoteVolume"]), r["symbol"]) for r in rows
           if r["symbol"].endswith("USDT") and r.get("quoteVolume")]
    return [s_ for _, s_ in sorted(liq, reverse=True)[:top]]


def listed_after(pair: str, ym: str = "2025-01") -> bool:
    """Cheap test for 'listed after the cut-off': no bar file in ``ym`` means listed later."""
    for probe in (ym, "2024-01", "2022-01", "2019-01"):
        url = f"{VISION}/spot/monthly/klines/{pair}/1d/{pair}-1d-{probe}.zip"
        if _month_zip_to_df(url, f"binance/probe/{pair}-1d-{probe}.zip") is not None:
            return False
    return True


def new_listings(top: int = 400, limit: int = 60) -> list[str]:
    out = []
    for pair in universe(top):
        if pair in MAJORS or pair in PEGS:
            continue
        try:
            if listed_after(pair, "2025-01"):
                name = spot_klines(pair, "1d", "2025-01")
                if name and not str(name).startswith(("too-few",)):
                    out.append(name)
        except Exception as e:                                         # noqa: BLE001
            record_failure("binance_new", pair, str(e))
        if len(out) >= limit:
            break
    return out


def first_listing_month(pair: str) -> str | None:
    """Cheapest reliable listing date: the first monthly 1d file that exists."""
    for y in range(2017, pd.Timestamp.now().year + 1):
        for m in range(1, 13):
            ym = f"{y}-{m:02d}"
            url = f"{VISION}/spot/monthly/klines/{pair}/1d/{pair}-1d-{ym}.zip"
            if _month_zip_to_df(url, f"binance/probe/{pair}-1d-{ym}.zip") is not None:
                return ym
    return None


# ------------------------------------------------------------ derivatives ---
def funding_history(pair: str, start_ms: int) -> pd.DataFrame:
    """USDT-M funding rate history (8 h steps). Not in any public TS corpus -> 'rare'."""
    rows, cur = [], start_ms
    while True:
        url = (f"{FUT_API}/fundingRate?symbol={pair}&startTime={cur}"
               f"&endTime={int(pd.Timestamp.now().timestamp()*1000)}&limit=1000")
        try:
            batch = json.loads(_get(url, 60))
        except Exception as e:                                       # noqa: BLE001
            record_failure("binance_funding", pair, str(e))
            break
        if not batch:
            break
        rows += batch
        nxt = int(batch[-1]["fundingTime"]) + 1
        if nxt <= cur or len(batch) < 1000:
            break
        cur = nxt
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True)
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df["mark_price"] = pd.to_numeric(df["markPrice"], errors="coerce")
    return df[["timestamp", "funding_rate", "mark_price"]].dropna(subset=["funding_rate"])


def _ts(ms) -> pd.Timestamp:
    return pd.to_datetime(int(ms), unit="ms", utc=True)


def funding_series(pair: str, since: str = "2020-01-01") -> str:
    df = funding_history(pair, int(pd.Timestamp(since).timestamp() * 1000))
    if len(df) < 200:
        return "too-few"
    name = f"cd_{pair.lower()}_funding"
    if dataset_exists(name):
        return "exists"
    write_open_dataset(name, df, freq="8h", domain="crypto_derivatives", task="forecast",
                       source=f"binance:futures:fundingRate:{pair}", license_="binance-tos",
                       target="funding_rate",
                       notes="USDT-M perpetual funding rate + mark price",
                       extra_meta={"contamination": "rare", "pair": pair,
                                   "symbol": pair, "listed_at": since})
    return name


def futures_klines(pair: str, interval: str = "1h", start: str = "2020-01") -> str:
    name = f"cf_{pair.lower()}_{interval}"
    if dataset_exists(name):
        return "exists"
    df = _kline_frame(pair, interval, start, market="futures/um")
    if len(df) < 500:
        return "too-few"
    write_open_dataset(name, df, freq=interval, domain="crypto_derivatives", task="forecast",
                       source=f"binance:futures:klines:{pair}:{interval}",
                       license_="binance-tos", target="close",
                       notes="USDT-M perpetual klines (last price)",
                       extra_meta={"contamination": "rare", "pair": pair, "symbol": pair})
    return name


def spot_klines(pair: str, interval: str = "1d", start: str = "2019-01",
                name_prefix: str = "cx") -> str:
    name = f"{name_prefix}_{pair.lower()}_{interval}"
    if dataset_exists(name):
        return "exists"
    df = _kline_frame(pair, interval, start)
    if len(df) < 500:
        return "too-few"
    listed = first_listing_month(pair) or ""            # one probe per pair, then cached
    if listed and listed >= CUTOFF[:7]:
        contamination = "post_listing"        # listed after the corpus cut-off: clean
    elif interval in ("1m", "5m", "15m") or str(df.timestamp.min())[:10] >= "2025-06-01":
        contamination = "fresh"               # post-dates every released model
    else:
        contamination = "public_risky"        # likely inside pre-training corpora
    write_open_dataset(name, df, freq=interval, domain="crypto", task="forecast",
                       source=f"binance:spot:{pair}:{interval}", license_="binance-tos",
                       target="close", notes=f"spot OHLCV + taker flow for {pair}",
                       extra_meta={"pair": pair, "listed_at": listed,
                                   "contamination": contamination})
    return name


def fresh_bars(pairs: list[str], interval: str = "5m", days: int = 75) -> list[str]:
    """Last ``days`` of fine bars: nobody's pretrained model has seen these."""
    out = []
    today = pd.Timestamp.now().normalize()            # naive: mixing tz-aware/naive in
    start = today - pd.Timedelta(days=days)           # date_range raises TypeError
    for pair in pairs:
        frames = []
        for d in [str(x.date()) for x in pd.date_range(start, today)]:
            url = f"{VISION}/spot/daily/klines/{pair}/{interval}/{pair}-{interval}-{d}.zip"
            df = _month_zip_to_df(url, f"binance/daily/{pair}/{interval}/{pair}-{interval}-{d}.zip")
            if df is not None:
                df["timestamp"] = _ts_col(df["open_time"])
                frames.append(df)
        if not frames:
            record_failure("binance_fresh", pair, f"no {interval} files in the window")
            continue
        df = pd.concat(frames, ignore_index=True)
        for c in ("open", "high", "low", "close", "volume", "quote_volume", "trades"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close"]).sort_values("timestamp")
        df = df.drop_duplicates(subset="timestamp")
        name = f"cx_{pair.lower()}_{interval}_fresh"
        write_open_dataset(name, df.reset_index(drop=True), freq=interval, domain="crypto",
                           source=f"binance:spot:{pair}:{interval}:last{days}d",
                           license_="binance-tos",
                           notes=f"post-cutoff fine bars ({start} ->)",
                           extra_meta={"contamination": "fresh", "pair": pair})
        out.append(name)
    return out


# ----------------------------------------------------------------- driver ---
def run(what: set[str], workers: int = 6, pairs: list[str] | None = None) -> dict:
    rep: dict[str, object] = {}
    pool = ThreadPoolExecutor(max_workers=workers)
    uni = pairs or MAJORS
    if "pegs" in what:                                   # small + serial, do them first
        rep["pegs"] = [spot_klines(p, "1d", "2019-01", name_prefix="peg") for p in PEGS]
    if "spot_daily" in what:
        rep["spot_daily"] = list(pool.map(lambda p: spot_klines(p, "1d", "2019-01"), uni))
    if "spot_1h" in what:
        rep["spot_1h"] = list(pool.map(lambda p: spot_klines(p, "1h", "2019-01"),
                                       uni[:12] if not pairs else uni))
    if "fut_klines" in what:
        rep["fut_klines"] = list(pool.map(lambda p: futures_klines(p, "1h"), uni[:12]))
    if "funding" in what:
        rep["funding"] = list(pool.map(funding_series, uni[:16]))
    if "new" in what:                    # listed after the corpus cut-off: zero contamination
        rep["new"] = new_listings()
    if "fresh" in what:
        rep["fresh"] = [(p, fresh_bars([p], iv, 75))
                        for p, iv in [(x, "5m") for x in uni[:10]]]
    rep = {k: (v if not isinstance(v, list) else [str(x) for x in v]) for k, v in rep.items()}
    pool.shutdown(wait=False)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--what", default="pegs,spot_daily,funding,fut_klines,fresh,spot_1h")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--pairs", default="", help="override the pair list")
    a = ap.parse_args()
    print(json.dumps(run(set(a.what.split(",")), a.workers,
                         a.pairs.split(",") if a.pairs else None), indent=1)[:4000])


if __name__ == "__main__":
    main()
