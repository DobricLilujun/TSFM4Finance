"""Assemble the standard-bench time-series data into the unified
``assets/data/standard/`` layout (one TSDataset per series).

Reuses tsfm_eval.register_dataset (bucket="standard"). No data is fabricated
or filled; splits come from the source per-split files (FX/FI) or a date-based
split (equity). Each dataset is a single series: timestamp + target column.
"""
from __future__ import annotations

import pathlib

import pandas as pd

from tsfm_eval import register_dataset
from tsfm_eval.schemas import DatasetMeta, Domain, Frequency, TaskType

ROOT = pathlib.Path(__file__).resolve().parent.parent
STDBENCH = ROOT / "assets" / "data" / "standard-bench"
FX = STDBENCH / "fx_data" / "processed"
FI = STDBENCH / "fixedincome_data" / "processed"
EQ = STDBENCH / "equity_data"

# ---- split date ranges (train 2013-2018 / val 2019-2020 / test 2021-2023) ----
TRAIN_MAX = pd.Timestamp("2018-12-31")
VAL_MIN, VAL_MAX = pd.Timestamp("2019-01-01"), pd.Timestamp("2020-12-31")
TEST_MIN = pd.Timestamp("2021-01-01")


def _ts(df: pd.DataFrame, col: str) -> pd.Series:
    """Normalize a timestamp column to naive datetime."""
    t = pd.to_datetime(df[col])
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_localize(None)
    return t


def _series(df: pd.DataFrame, ts: str, target: str, sort_by: str = "timestamp") -> pd.DataFrame:
    """Keep only timestamp + target, drop NaN target, sort chronologically."""
    out = df[[ts, target]].copy()
    out = out.dropna(subset=[target])
    out = out.sort_values(ts).reset_index(drop=True)
    out[ts] = _ts(out, ts)
    out.columns = ["timestamp", target]
    return out


def _meta(name, domain, freq, target, lookback, horizon, source, notes="") -> DatasetMeta:
    return DatasetMeta(
        name=name, domain=domain, task=TaskType.FORECAST, frequency=freq,
        n_features=0, feature_names=[], target=target,
        lookback=lookback, horizon=horizon, rolling=True,
        license="open", source=source, open=True, requires_download=False,
        notes=notes,
    )


def _register(name, domain, freq, target, lb, hz, train, val, test, source, notes=""):
    meta = _meta(name, domain, freq, target, lb, hz, source, notes)
    register_dataset(name, meta, train, val, test, bucket="standard")


# --------------------------------------------------------------------------- #
# FX (daily + hourly): per pair, target = close
# ---------------------------------------------------------------------------
def build_fx():
    lb = {"day": (52, 10), "hour": (96, 24)}
    for freq, prefix in [("day", "fx_1d"), ("hour", "fx_1h")]:
        files = {s: pd.read_csv(FX / f"{prefix}_{s}.csv.gz") for s in
                 ("train", "validation", "test")}
        val_pairs = sorted(files["validation"]["pair"].unique())
        built = 0
        for pair in val_pairs:
            sub = {}
            for s in ("train", "validation", "test"):
                d = files[s][files[s]["pair"] == pair]
                sub[s] = _series(d, "timestamp", "close")
            # only keep series that have validation data
            if len(sub["validation"]) == 0:
                continue
            name = f"std_fx_{prefix[3:]}_{pair.lower()}"
            _register(name, Domain.FX, freq, "close", lb[freq][0], lb[freq][1],
                      sub["train"], sub["validation"], sub["test"],
                      source=f"Dukascopy/Yahoo {pair} {prefix}",
                      notes=f"FX {pair} {freq} level (close); {len(sub['validation'])} val obs")
            built += 1
        print(f"  FX {freq}: {built} datasets")


# --------------------------------------------------------------------------- #
# Fixed income: per (country, maturity), target = yield_pct
# ---------------------------------------------------------------------------
def build_fi():
    files = {s: pd.read_csv(FI / f"yields_{s}.csv.gz") for s in
             ("train", "validation", "test")}
    combos = sorted(files["validation"].itertuples(index=False),
                    key=lambda r: (r.country, r.maturity_years))
    seen = set()
    built = 0
    for row in combos:
        key = (row.country, int(row.maturity_years))
        if key in seen:
            continue
        seen.add(key)
        sub = {}
        for s in ("train", "validation", "test"):
            d = files[s][(files[s]["country"] == row.country) &
                         (files[s]["maturity_years"] == key[1])]
            sub[s] = _series(d, "date", "yield_pct")
        if len(sub["validation"]) == 0:
            continue
        name = f"std_fi_1d_{row.country.lower()}_{key[1]}y"
        _register(name, Domain.BOND, "day", "yield_pct", 52, 10,
                  sub["train"], sub["validation"], sub["test"],
                  source=f"{row.source} {row.country} {key[1]}y",
                  notes=f"Yield {row.country} {key[1]}y; {len(sub['validation'])} val obs")
        built += 1
    print(f"  FI: {built} datasets")


# --------------------------------------------------------------------------- #
# Equity: per permno, target = split_adjusted_close, date-based split
# ---------------------------------------------------------------------------
def build_equity():
    frames = [pd.read_csv(EQ / f"daily_ohlcv_{y}.csv.gz") for y in range(2013, 2024)]
    df = pd.concat(frames, ignore_index=True)
    df["date"] = _ts(df, "date")
    df = df.dropna(subset=["split_adjusted_close"])
    df = df.sort_values(["permno", "date"]).reset_index(drop=True)
    built = 0
    for permno, g in df.groupby("permno"):
        g = g.sort_values("date").reset_index(drop=True)
        train = g[g["date"] <= TRAIN_MAX][["date", "split_adjusted_close"]]
        val = g[(g["date"] >= VAL_MIN) & (g["date"] <= VAL_MAX)][["date", "split_adjusted_close"]]
        test = g[g["date"] >= TEST_MIN][["date", "split_adjusted_close"]]
        if len(val) == 0:
            continue
        train.columns = ["timestamp", "split_adjusted_close"]
        val.columns = ["timestamp", "split_adjusted_close"]
        test.columns = ["timestamp", "split_adjusted_close"]
        name = f"std_equity_1d_{int(permno)}"
        _register(name, Domain.EQUITY, "day", "split_adjusted_close", 52, 10,
                  train, val, test,
                  source=f"equity permno {int(permno)}",
                  notes=f"Equity {int(permno)} split-adj close; {len(val)} val obs")
        built += 1
    print(f"  Equity: {built} datasets")


def main():
    print("Building standard-bench datasets -> assets/data/standard/")
    build_fx()
    build_fi()
    build_equity()
    print("Done.")


if __name__ == "__main__":
    main()
