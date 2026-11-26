#!/usr/bin/env python3
"""Repair non-chronological / duplicated splits in ``data/open/<dataset>/``.

Why this exists (found by scripts/audit_data.py on 2026-11-11): the 12 legacy daily series
(``sp500_daily`` & co) were split by taking the first 70 % of a frame that was **not** sorted
by date.  The result:

    sp500_daily  train 2023-08-15 -> 2026-03-13   (647 rows)
                 val   2025-09-22 -> 2026-03-04   (113 rows)
                 test  2026-03-05 -> 2026-08-14   (113 rows)

i.e. ``train`` contains 120 rows that also sit in val/test, and it runs *past* the start of
the test window.  Anything fitted on that train split has seen the evaluation period, so
every number computed on those 12 series (the §4.4 case-study table) is contaminated and has
to be regenerated after the repair.

The repair is content-preserving: rows are de-duplicated on ``timestamp`` (keeping the first
occurrence), sorted, and re-split 70/15/15 chronologically.  Columns, values and the date
range are untouched.  Re-run ``--stage derive`` afterwards so the ``_dir5/_volreg/_extreme/
_trade`` views inherit the corrected boundary.

  .venv2/bin/python scripts/repair_splits.py [--dry-run] [--only name1,name2] [--all]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OPEN = REPO / "data" / "open"
MANIFEST = REPO / "data" / "manifest.json"
SPLIT = (0.70, 0.15, 0.15)
DERIVED = ("_dir5", "_volreg", "_extreme", "_trade")


def load_all(d: Path) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(d / f"{s}.parquet") for s in
                      ("train", "validation", "test")], ignore_index=True)


def chronological(full: pd.DataFrame) -> bool:
    ts = pd.to_datetime(full.timestamp)
    return not ts.duplicated().any()


def repair(name: str, dry: bool) -> str:
    d = OPEN / name
    meta_path = d / "meta.json"
    meta = json.loads(meta_path.read_text())
    full = load_all(d)
    ts = pd.to_datetime(full.timestamp)
    n_dup = int(ts.duplicated().sum())
    ordered = ts.sort_values()
    spans_touch = bool((pd.to_datetime(pd.read_parquet(d / "train.parquet").timestamp).max()
                        >= pd.to_datetime(pd.read_parquet(d / "test.parquet").timestamp).min()))
    if n_dup == 0 and not spans_touch:
        return "clean"
    if dry:
        return f"would-repair dup={n_dup} rows={len(full)}"
    full["timestamp"] = pd.to_datetime(full["timestamp"])
    full = full.dropna(subset=[meta.get("target") or "close"], how="all")
    full = full.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="first")
    full = full.reset_index(drop=True)
    n = len(full)
    n_train, n_val = int(n * SPLIT[0]), int(n * SPLIT[1])
    full.iloc[:n_train].to_parquet(d / "train.parquet")
    full.iloc[n_train:n_train + n_val].to_parquet(d / "validation.parquet")
    full.iloc[n_train + n_val:].to_parquet(d / "test.parquet")
    meta["n_obs"] = n
    meta["split"] = {"train": n_train, "validation": n_val, "test": n - n_train - n_val}
    meta["date_range"] = [str(full.timestamp.min())[:10], str(full.timestamp.max())[:10]]
    meta["notes"] = (str(meta.get("notes", "")) +
                     " | splits rebuilt chronologically 2026-11-11 (the original train split "
                     "overlapped val/test; see scripts/repair_splits.py)").strip()
    meta_path.write_text(json.dumps(meta, indent=1))
    if MANIFEST.exists():                      # keep the checksum record truthful
        man = json.loads(MANIFEST.read_text())
        if name in man:
            import hashlib
            h = hashlib.sha256()
            with open(d / "test.parquet", "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            man[name].update(n_obs=n, date_range=meta["date_range"], sha256=h.hexdigest()[:16],
                             updated_at="repaired-2026-11-11")
            MANIFEST.write_text(json.dumps(man, indent=1))
    return f"repaired dup={n_dup} -> {n} rows"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--all", action="store_true", help="include derived task datasets")
    a = ap.parse_args()
    names = [p.name for p in sorted(OPEN.iterdir()) if (p / "meta.json").exists()]
    if a.only:
        names = a.only.split(",")
    if not a.all:
        names = [n for n in names if not n.endswith(DERIVED)]
    changed = []
    for n in names:
        try:
            r = repair(n, a.dry_run)
        except Exception as e:                                      # noqa: BLE001
            print(f"  {n:<34} ERROR {e}")
            continue
        if r not in ("clean",):
            changed.append(n)
            print(f"  {n:<34} {r}")
    print(("DRY RUN — " if a.dry_run else "") + f"{len(changed)} dataset(s) need repair")
    if changed and not a.dry_run:
        print("re-run the derived views:  "
              ".venv2/bin/python -m tsfm4finance.bench.sweep --stage derive --groups <groups>")
    sys.exit(0)


if __name__ == "__main__":
    main()
