#!/usr/bin/env python3
"""Materialise `timestamp,close` CSVs from the corpus's parquet splits.

`sweep_evaluate.py --series` takes a CSV with `timestamp,close`, while the benchmark
corpus stores each series as train/validation/test parquet under
`assets/data/<group>/<name>/`. Rather than teach the sweep about the corpus layout (and
give every machine a different code path), flatten the splits into the shape the sweep
already documents.

This is not a leak and not a protocol change: the zero-shot protocol slides windows over
`--spans all`, which means it never uses split boundaries as held-out data — the split
columns are ignored on purpose. When scoring *adapted* checkpoints the sweep is called
with `--spans test`, and then the split boundary has to come from a real split, so a
flattened CSV must not be used for that. The note is written into the sidecar so a table
cannot silently mix the two.

    series_to_csv.py --list pass2_trux.txt --out reports/trux_pass2/series_csv
"""
from __future__ import annotations

import argparse
import json
import pathlib

import pandas as pd

SPLITS = ("train", "validation", "test")


def flatten(d: pathlib.Path) -> tuple[pd.DataFrame, dict]:
    frames = []
    for sp in SPLITS:
        f = d / f"{sp}.parquet"
        if f.exists():
            frames.append((sp, pd.read_parquet(f)))
    if not frames:
        raise SystemExit(f"no {SPLITS} parquet under {d}")
    ts = "timestamp" if "timestamp" in frames[0][1].columns \
        else frames[0][1].columns[0]
    tgt = "close" if "close" in frames[0][1].columns \
        else [c for c in frames[0][1].columns if c != ts][0]

    # per-split ranges, because the corpus is not uniform: `standard/*` splits are
    # disjoint and ordered, `open/*` splits carry a lookback context that repeats the
    # previous split's tail (by convention), and at least one `open/*` series has a train
    # range that runs *past* the validation range. Flattening without recording that lets
    # a zero-shot table and a fine-tune table claim the same rows mean the same thing.
    ranges, overlap = {}, "none"
    for sp, df in frames:
        t = pd.to_datetime(df[ts])
        ranges[sp] = (str(t.min().date()), str(t.max().date()))
    seen, first_of = {}, {}
    for sp, df in frames:
        for v in pd.to_datetime(df[ts]):
            first_of.setdefault(v, sp)
        seen[sp] = set(pd.to_datetime(df[ts]))
    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    shared = {f"{a}\u2229{b}": len(seen[a] & seen[b]) for a, b in pairs
              if a in seen and b in seen}

    out = pd.concat([df[[ts, tgt]].rename(columns={ts: "timestamp", tgt: "close"})
                     for _, df in frames], ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.sort_values("timestamp").reset_index(drop=True)

    # the only acceptable duplicate is the same observation repeated with the same value
    dup = out[out["timestamp"].duplicated(keep=False)]
    if len(dup):
        conflict = (dup.groupby("timestamp")["close"].nunique() > 1)
        if conflict.any():
            raise SystemExit(f"{d.name}: {int(conflict.sum())} timestamps carry different "
                             f"values across splits - refusing to guess which is real")
    raw = len(out)
    out = out.drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)
    if any(v for v in shared.values()):
        overlap = ("non-monotone split ranges" if ranges.get("train", ("0", "0"))[1]
                   > ranges.get("validation", ("0", "9999"))[1] and shared.get(
                       "train\u2229validation") else "context reuse")
    return out, {"rows_raw": raw, "rows_deduped": int(len(out)),
                 "dropped_duplicates": raw - int(len(out)),
                 "split_overlap": shared, "split_ranges": ranges,
                 "overlap_kind": overlap}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="file of dataset dirs, one per line")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in pathlib.Path(a.list).read_text().splitlines():
        d = pathlib.Path(line.strip())
        if not d.is_dir():
            print(f"skip (not a dir): {d}")
            continue
        df, meta = flatten(d)
        p = out / f"{d.name}.csv"
        df.to_csv(p, index=False)
        rows.append({"series": d.name, "csv": str(p), "source_dir": str(d), **meta,
                     "start": str(df.timestamp.iloc[0]), "end": str(df.timestamp.iloc[-1])})
        print(f"  {d.name:26s} {meta['rows_deduped']:>7} rows "
              f"(-{meta['dropped_duplicates']} dup, {meta['overlap_kind']})")
    flagged = [r["series"] for r in rows if r["overlap_kind"] == "non-monotone split ranges"]
    (out / "MANIFEST.json").write_text(json.dumps({
        "protocol_note": "flattened splits; for zero-shot --spans all only; adapted "
                         "checkpoints need a real split and --spans test",
        "duplicates": "same timestamp with the same value, kept first; a timestamp "
                      "carrying different values aborts the run",
        "flagged_non_monotone_splits": flagged, "series": rows}, indent=1))
    print(f"{len(rows)} csv written; manifest at {out / 'MANIFEST.json'}")
    if flagged:
        print("WARNING: non-monotone split ranges in: " + ", ".join(flagged)
              + " — zero-shot rows are fine, but do not use these as held-out splits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
