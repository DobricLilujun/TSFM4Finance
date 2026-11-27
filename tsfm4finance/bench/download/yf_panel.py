"""T1 + T2 equity/index/FX/commodity/crypto panels from Yahoo Finance.

Two deliberate upgrades over the original 3-year ``data/raw`` pull:

* **OHLCV columns** (open/high/low/close/volume).  Without them the finance-native
  TSFM (Kronos, which consumes K-line tokens) cannot be evaluated at all, and no
  candlestick/intraday task can be built.
* **Long history** (2004 or the asset's first listing -> today).  The 3-year panel
  gives 5-8 non-overlapping windows per series at h=21, which cannot separate
  models at the 0.1-MASE level and covers a single regime; the val-vs-test control
  already showed rankings move between spans.

A STOOQ fallback is included because Yahoo truncates some assets (index history,
commodity futures), and STOOQ publishes plain CSVs with decades of daily data.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import requests

from .common import (OPEN, PANELS, RAW, dataset_exists, http_get,
                     record_failure, clear_failure, retry, write_open_dataset)

# (yahoo ticker, dataset name, domain, start date, long-history note)
LONG12: list[tuple[str, str, str, str]] = [
    ("^GSPC", "sp500_long", "index", "2004-01-01"),
    ("^IXIC", "nasdaq_long", "index", "2004-01-01"),
    ("^DJI", "dow_long", "index", "2004-01-01"),
    ("^RUT", "russell2000_long", "index", "2004-01-01"),
    ("BTC-USD", "btc_long", "crypto", "2014-01-01"),
    ("ETH-USD", "eth_long", "crypto", "2015-08-01"),
    ("EURUSD=X", "eurusd_long", "fx", "2004-01-01"),
    ("GBPUSD=X", "gbpusd_long", "fx", "2004-01-01"),
    ("GC=F", "gold_long", "commodity", "2004-01-01"),
    ("CL=F", "crude_oil_long", "commodity", "2004-01-01"),
    ("IEF", "us_treasury_7y_long", "rates", "2004-01-01"),
    ("LQD", "ig_corp_bond_long", "credit", "2007-01-01"),
]

NEWX: list[tuple[str, str, str, str]] = [
    ("^VIX", "vix", "volatility", "2004-01-01"),
    ("^TNX", "us_10y_yield", "rates", "2004-01-01"),
    ("^TYX", "us_30y_yield", "rates", "2004-01-01"),
    ("^FVX", "us_5y_yield", "rates", "2004-01-01"),
    ("^IRX", "us_13w_bill", "rates", "2004-01-01"),
    ("DX-Y.NYB", "dxy", "fx", "2004-01-01"),
    ("JPY=X", "usdjpy", "fx", "2004-01-01"),
    ("AUDUSD=X", "audusd", "fx", "2004-01-01"),
    ("CHF=X", "usdchf", "fx", "2004-01-01"),
    ("CNY=X", "usdcny", "fx", "2010-01-01"),
    ("BZ=F", "brent", "commodity", "2010-01-01"),
    ("HG=F", "copper", "commodity", "2007-01-01"),
    ("NG=F", "nat_gas", "commodity", "2010-01-01"),
    ("ZW=F", "wheat", "commodity", "2010-01-01"),
    ("ZS=F", "soybean", "commodity", "2010-01-01"),
    ("SI=F", "silver", "commodity", "2007-01-01"),
    ("^GDAXI", "dax", "index", "2004-01-01"),
    ("^FTSE", "ftse100", "index", "2004-01-01"),
    ("^FCHI", "cac40", "index", "2004-01-01"),
    ("^N225", "nikkei225", "index", "2004-01-01"),
    ("^HSI", "hang_seng", "index", "2004-01-01"),
    ("000001.SS", "shanghai_composite", "index", "2004-01-01"),
    ("^STOXX50E", "eurostoxx50", "index", "2004-01-01"),
    ("2800.HK", "etracs_china", "index", "2007-01-01"),
    ("TLT", "us_20y_etf", "rates", "2007-01-01"),
    ("HYG", "hy_credit_etf", "credit", "2007-01-01"),
    ("QQQ", "nasdaq_etf", "equity_etf", "2004-01-01"),
    ("IWM", "russell_etf", "equity_etf", "2004-01-01"),
    ("XLK", "sector_technology", "equity_etf", "2004-01-01"),
    ("XLE", "sector_energy", "equity_etf", "2004-01-01"),
    ("XLF", "sector_financials", "equity_etf", "2004-01-01"),
    ("GLD", "gold_etf", "commodity", "2004-01-01"),
    ("DBC", "commodity_index_etf", "commodity", "2007-01-01"),
]

# T2: cross-sectional panel.  Constituent list is scraped (see constituents());
# this fallback list keeps the run alive when the scrape is blocked.
NDAQ100_FALLBACK = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "AVGO", "TSLA", "COST", "ADBE", "AMD",
    "INTC", "QCOM", "CSCO", "TXN", "MU", "NFLX", "PEP", "CMCSA", "INTU", "AMGN",
    "ISRG", "SBUX", "AMAT", "GILD", "BKNG", "ADP", "MDLZ", "VRTX", "REGN", "LRCX",
    "PANW", "SNPS", "CDNS", "MELI", "FTNT", "KLAC", "PYPL", "MAR", "ORLY", "CSX",
    "ADI", "MRVL", "NXPI", "ABNB", "BX", "ROST", "MCHP", "TRI", "PLTR", "LULU",
    "AZN", "CRWD", "WDAY", "TTD", "DECK", "FANG", "PCAR", "CCEP", "CHTR", "DASH",
    "ADBE2", "IDXX", "EXC", "XEL", "PDD", "ANSS", "DLTR", "SWKS", "INSM", "ENPH",
    "MNST", "CTSH", "ORPX", "SLAB", "ROST2", "TEAM", "WBD", "ZS", "APP", "GEHC",
]


@retry(times=3, base_delay=2.0)
def _yf(ticker: str, start: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(ticker, start=start, interval="1d", progress=False,
                     auto_adjust=True, threads=False)
    if df is None or len(df) < 200:
        raise RuntimeError(f"too few rows ({0 if df is None else len(df)})")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    first = df.columns[0]
    df = df.rename(columns={first: "timestamp"})
    keep = {}
    for src, dst in (("Open", "open"), ("High", "high"), ("Low", "low"),
                     ("Close", "close"), ("Adj Close", "adj_close"),
                     ("Volume", "volume")):
        if src in df.columns:
            keep[dst] = pd.to_numeric(df[src], errors="coerce")
    out = pd.DataFrame({"timestamp": pd.to_datetime(df["timestamp"])})
    out = pd.concat([out, pd.DataFrame(keep)], axis=1)
    return out.dropna(subset=["close"]).reset_index(drop=True)


@retry(times=2, base_delay=1.5)
def _stooq(symbol: str) -> pd.DataFrame:
    """STOOQ publishes plain daily CSVs, often with longer history than Yahoo."""
    body = http_get(f"https://stooq.com/q/d/l/?s={symbol}&i=d", timeout=60).decode("utf-8", "replace")
    df = pd.read_csv(io.StringIO(body))
    if df.shape[0] < 200 or "Date" not in df.columns:
        raise RuntimeError(f"stooq gave {df.shape[0]} rows")
    df = df.rename(columns={"Date": "timestamp", "Open": "open", "High": "high",
                            "Low": "low", "Close": "close", "Volume": "volume"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.dropna(subset=["close"]).reset_index(drop=True)


STOOQ_FALLBACK = {
    "sp500_long": "^spx", "nasdaq_long": "^ndq", "dow_long": "^dji",
    "russell2000_long": "^rut", "eurusd_long": "eurusd", "gbpusd_long": "gbpusd",
    "gold_long": "xauusd", "crude_oil_long": "cl.f", "vix": "^vix",
    "dxy": "dx.f", "brent": "cb.f", "copper": "hg.f", "nikkei225": "^nkx",
    "hang_seng": "^hsi", "dax": "^dax", "ftse100": "^ukx",
    "shanghai_composite": "^shc", "dxy": "usdx", "nikkei225": "^nky",
}


def fetch_one(ticker: str, name: str, domain: str, start: str, *, force: bool = False,
              stooq_symbol: str | None = None) -> str:
    """Return 'ok' | 'exists' | 'failed:<reason>'."""
    if dataset_exists(name) and not force:
        return "exists"
    df, src = None, f"yahoo:{ticker}"
    try:
        df = _yf(ticker, start)
    except Exception as e:                                # noqa: BLE001
        fb = stooq_symbol or STOOQ_FALLBACK.get(name)
        if fb:
            try:
                                df, src = _stooq(fb), f"stooq:{fb}"
            except Exception as e2:                       # noqa: BLE001
                record_failure("yf_panel", name, f"yahoo: {e} | stooq: {e2}")
                return f"failed:yahoo={e};stooq={e2}"
        else:
            record_failure("yf_panel", name, str(e))
            return f"failed:{e}"
    if df is None or len(df) < 200:
        record_failure("yf_panel", name, "empty")
        return "failed:empty"
    write_open_dataset(
        name, df, freq="day", domain=domain, source=src, license_="market-data terms",
        notes=f"OHLCV daily, {start} -> present; auto-adjusted closes; "
              f"chronological 70/15/15 splits",
        extra_meta={"ticker": ticker, "ohlcv": True, "start_requested": start},
    )
    clear_failure("yf_panel", name)
    return "ok"


def run_long12(force: bool = False) -> dict:
    return {name: fetch_one(t, name, dom, start, force=force)
            for t, name, dom, start in LONG12}


def run_newx(force: bool = False) -> dict:
    return {name: fetch_one(t, name, dom, start, force=force)
            for t, name, dom, start in NEWX}


# --------------------------------------------------------------------------- #
def constituents(source: str = "nasdaq100") -> list[str]:
    """Point-in-time-ish constituent list from the Wikipedia category table.

    This is a *current* snapshot, so a cross-sectional backtest on it carries
    survivorship bias; the benchmark report states that instead of hiding it.
    """
    url = {"nasdaq100": "https://en.wikipedia.org/wiki/Nasdaq-100",
           "dow30": "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
           "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"}[source]
    try:
        tables = pd.read_html(url)
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            tick_col = next((c for c, lc in zip(t.columns, cols)
                             if "ticker" in lc or lc.strip() == "symbol"), None)
            if tick_col is None:
                continue
            out = [str(x).strip().replace(".", "-") for x in t[tick_col].dropna()]
            out = [x for x in out if x and (x.isalnum() or "-" in x)]
            if len(out) >= 20:
                return sorted(set(out))
    except Exception:                                     # noqa: BLE001
        pass
    return sorted(set(NDAQ100_FALLBACK))


def run_cross_section(source: str = "nasdaq100", start: str = "2010-01-01",
                      limit: int = 0, force: bool = False,
                      max_names: int = 110) -> dict:
    """Download the constituent panel as individual OHLCV datasets plus one
    aligned wide panel used by the cross-sectional (ranking / top-k) tasks."""
    tickers = constituents(source)[:max_names]
    if limit:
        tickers = tickers[:limit]
    res: dict[str, str] = {}
    closes: dict[str, pd.Series] = {}
    for tk in tickers:
        name = f"xs_{tk.lower().replace('-', '_')}"
        res[tk] = fetch_one(tk, name, "equity", start, force=force)
        if res[tk] in ("ok", "exists"):
            d = OPEN / name
            try:
                fr = pd.concat([pd.read_parquet(d / f) for f in
                                ("train.parquet", "validation.parquet", "test.parquet")])
                closes[tk] = pd.Series(fr["close"].to_numpy(),
                                       index=pd.to_datetime(fr["timestamp"]))
            except Exception:                             # noqa: BLE001
                pass
    if closes:
        panel = pd.DataFrame(closes).sort_index()
        panel.index.name = "timestamp"
        panel.to_parquet(PANELS / f"{source}_daily_close.parquet")
        ret = panel.pct_change()
        ret.to_parquet(PANELS / f"{source}_daily_ret.parquet")
        (PANELS / f"{source}_meta.json").write_text(json.dumps({
            "source": source, "n_names": int(panel.shape[1]),
            "date_range": [str(panel.index.min())[:10], str(panel.index.max())[:10]],
            "caveat": "constituent list is a current snapshot -> survivorship bias; "
                      "cross-sectional numbers are reported with that caveat",
        }, indent=1))
    return res
