"""Download real financial data to data/raw/<ticker>.csv (robust, per-ticker timeout).

Isolates the slow network from arena/datasets.py, which reads this cache first.
Run:  uv run python -m arena.download_real
Each ticker is downloaded independently; a failure is skipped, not fatal.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"
ROOT.mkdir(parents=True, exist_ok=True)

# (ticker, name) — covers index / equity / crypto / fx / commodity / bond
TICKERS = [
    ("^GSPC", "sp500"),
    ("^IXIC", "nasdaq"),
    ("^DJI", "dow"),
    ("^RUT", "russell2000"),
    ("BTC-USD", "btc"),
    ("ETH-USD", "eth"),
    ("EURUSD=X", "eurusd"),
    ("GBPUSD=X", "gbpusd"),
    ("GC=F", "gold"),
    ("CL=F", "crude_oil"),
    ("IEF", "bond_3_7y"),
    ("LQD", "bond_ig_corp"),
]


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance 1.x returns index=Date, columns=MultiIndex(Price, Ticker).

    Collapse the (Price, Ticker) column index to the metric name (level 0),
    so columns become Close/High/Low/Open/Volume.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # keep the metric level (e.g. 'Close'); drop the ticker level
        df.columns = df.columns.get_level_values(0)
    # turn the Date index into a 'Date' column
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        # the reset column may be unnamed / 'Date' / 'Datetime'; normalize
        first = df.columns[0]
        if "Date" not in df.columns and str(first).lower() in ("datetime", "date", ""):
            df = df.rename(columns={first: "Date"})
    return df


def _pick_close(df: pd.DataFrame) -> pd.DataFrame | None:
    col = "Close" if "Close" in df.columns else None
    if col is None:
        price = [c for c in df.columns if any(k in str(c).lower()
                   for k in ("close", "adj", "price"))]
        col = price[0] if price else (df.columns[-1] if len(df.columns) else None)
    if col is None or "Date" not in df.columns:
        return None
    out = pd.DataFrame({"timestamp": df["Date"], "close": df[col]})
    out = out.dropna().reset_index(drop=True)
    return out if len(out) >= 120 else None


def main() -> None:
    ok, fail = [], []
    for ticker, name in TICKERS:
        t0 = time.time()
        try:
            df = yf.download(ticker, period="3y", interval="1d",
                            progress=False, auto_adjust=True,
                            threads=False)
            if df is None or len(df) < 120:
                raise RuntimeError("too few rows")
            df = _flatten(df)
            out = _pick_close(df)
            if out is None:
                raise RuntimeError("no close column")
            path = ROOT / f"{name}.csv"
            out.to_csv(path, index=False)
            ok.append((ticker, len(out)))
            print(f"  [ok] {ticker:12s} -> {name:14s} {len(out):4d} rows "
                  f"({time.time()-t0:4.1f}s)")
        except Exception as e:
            fail.append((ticker, str(e)))
            print(f"  [fail] {ticker:12s} {e} ({time.time()-t0:4.1f}s)")
        time.sleep(0.4)
    print(f"\nDownloaded {len(ok)}/{len(TICKERS)} -> {ROOT}")
    if fail:
        print("Failed:", [t for t, _ in fail])


if __name__ == "__main__":
    main()