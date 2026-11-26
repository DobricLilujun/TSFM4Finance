#!/usr/bin/env python3
"""Data acceptance audit for the Arena (docs/11_data_plan.md §5).

Checks, per dataset: row count vs. the resolution the horizon needs, duplicate/unordered
timestamps, calendar gaps (crypto trades 24/7 - a gap there is a download bug, while a
Sunday gap in equity data is correct), date range, target column present, and whether the
dataset is registered in data/manifest.json.

Run:  .venv2/bin/python scripts/audit_data.py [--groups crypto_spot,...]
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OPEN = REPO / "data" / "open"
DERIVED = ("_dir5", "_volreg", "_extreme", "_trade")
# expected median step in minutes, from the meta frequency label
STEP = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "8h": 480, "1d": 1440,
        "daily": 1440, "weekly": 10080, "monthly": 43200, "quarterly": 129600}
# markets that close: gaps are expected; crypto does not close
TRADING_MARKETS = ("index", "equity", "equity_etf", "bond", "credit", "commodity",
                   "fx", "rates", "volatility", "macro", "fraud", "general", "inflation",
                   "prices", "financial_conditions")
# what a series must have to be useful at h<=63 bars of its own frequency
MIN_OBS = {"1m": 3000, "5m": 3000, "15m": 2000, "30m": 2000, "1h": 1500, "8h": 1200,
           "1d": 400, "daily": 400, "weekly": 200, "monthly": 120, "quarterly": 60}
# vendors that publish trading-day series: weekend gaps are correct, not a download bug
TRADING_DAY_SOURCES = ("fred:", "yahoo:")


def check(name: str, man: dict) -> dict:
    d = OPEN / name
    meta = json.loads((d / "meta.json").read_text())
    out = {"name": name, "domain": meta.get("domain"), "freq": meta.get("frequency"),
           "n": meta.get("n_obs"), "target": meta.get("target"),
           "contamination": meta.get("contamination"), "range": None, "flags": []}
    parts = {p: pd.read_parquet(d / f"{p}.parquet") for p in ("train", "validation", "test")}
    full = pd.concat(parts.values(), ignore_index=True)
    ts = pd.to_datetime(full["timestamp"])
    out["range"] = [str(ts.min())[:10], str(ts.max())[:10]]
    if target := out["target"]:
        if target not in full.columns and "close" not in full.columns:
            out["flags"].append("target-column-missing")
    if ts.duplicated().any():
        out["flags"].append(f"duplicate-timestamps={int(ts.duplicated().sum())}")
    if not ts.is_monotonic_increasing:
        out["flags"].append("unsorted-within-file")
    # leakage / split sanity
    if len(parts["test"]) == 0 or len(parts["train"]) == 0:
        out["flags"].append("empty-split")
    if max(parts["train"].index) >= 0 and pd.to_datetime(parts["train"].timestamp).max() >= \
       pd.to_datetime(parts["test"].timestamp).min():
        out["flags"].append("split-overlap")
    # calendar gap audit
    step = STEP.get(str(out["freq"]).lower())
    if step and len(ts) > 50:
        med = ts.diff().dt.total_seconds().median() / 60.0
        if abs(med - step) > 0.5 * step:
            out["flags"].append(f"median-step-{round(med)}min-expected-{step}")
        src = str(json.loads((OPEN / name / "meta.json").read_text()).get("source", ""))
        if (str(out["domain"]) not in TRADING_MARKETS
                and not src.startswith(TRADING_DAY_SOURCES)):
            big = (ts.diff().dt.total_seconds() / 60 > 2.5 * step).sum()
            if big > 0.01 * len(ts):
                out["flags"].append(f"24/7-gaps={int(big)}")
    if name not in man:
        out["flags"].append("not-in-manifest")
    need = MIN_OBS.get(str(out["freq"]).lower(), 400)
    if out["n"] and out["n"] < need:
        out["flags"].append(f"short-series-{out['n']}<-{need}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="")
    ap.add_argument("--base-only", action="store_true")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    names = sorted(p.name for p in OPEN.iterdir() if (p / "meta.json").exists())
    if a.groups:
        import yaml
        sg = yaml.safe_load((REPO / "tsfm4finance" / "bench" / "config.yaml").read_text())
        keep = {n for g in a.groups.split(",") for n in sg["series_groups"].get(g, [])}
        names = [n for n in names if n in keep]
    if a.base_only:
        names = [n for n in names if not n.endswith(DERIVED)]
    man = json.loads((REPO / "data" / "manifest.json").read_text()) \
        if (REPO / "data" / "manifest.json").exists() else {}
    rows = [check(n, man) for n in names]
    bad = [r for r in rows if r["flags"]]
    by_dom = Counter((str(r["domain"]), str(r["contamination"]))
                     for r in rows if not r["name"].endswith(DERIVED))
    print(f"datasets checked: {len(rows)}  clean: {len(rows) - len(bad)}  flagged: {len(bad)}")
    print("\nby (domain, contamination), base datasets only:")
    for (dom, cont), k in sorted(by_dom.items()):
        print(f"  {str(dom):<20} {str(cont):<14} {k}")
    if bad:
        print("\nflagged:")
        for r in bad[:40]:
            print(f"  {r['name']:<34} n={str(r['n']):>7}  {', '.join(r['flags'])}")
    if len(bad) > 40:
        print(f"  ... {len(bad) - 40} more")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
