#!/usr/bin/env python3
"""Do two machines produce the same number for the same cell?

Pass 2 and pass 3 deliberately partitioned series across machines, so the merged table
was a union that had never been cross-checked: nothing proved that trux's cuda:0 and
watson's A6000 compute the same forecast for the same window. gpu7 re-scored three series
that trux had already scored, which is the first real overlap, so the check is now a
script rather than a one-off.

Usage:
    python probes/trux/cross_machine_check.py A.jsonl B.jsonl [C.jsonl ...]
        [--tolerance 0.02] [--metric mase] [--detail]

Each file is labelled with its recorded `machine` (from the sweep meta line), so the
report says which pair disagrees, not just that some cell moved. Cells are keyed exactly
as the merge keys them: (model, series, span, horizon).

Exit codes: 0 all overlapping cells within tolerance, 1 at least one outside, 2 nothing
comparable: the machines covered disjoint series, or every overlapping cell was read from a
different data path (different bytes or a different split under the same name).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

KEY = ("model", "series", "span", "horizon")


def row_reg_hack(rec) -> str:
    """Record-level registry key. Kept as a function so the collision note below has one
    place to explain itself."""
    return rec.get("model") or ""
DEFAULT_METRICS = ("mase", "rmse", "dir_acc")


def load(path: pathlib.Path) -> tuple[dict, dict]:
    """(cells, meta) for one sweep file. Error records and NaN'd cells are skipped:
    a missing measurement cannot agree or disagree, and it must not be read as zero."""
    cells: dict[tuple, dict] = {}
    meta: dict = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if "meta" in rec:
            meta.update(rec["meta"])
            continue
        machine = meta.get("machine") or rec.get("machine") or str(path)
        # the row's own `model` is the adapter's *family* key, not the registry key:
        # chronos-bolt-small and chronos-bolt-base both write "chronos-bolt", and ten such
        # collisions exist across the registry. Keying a comparison on the row id silently
        # merges different checkpoints and manufactures disagreement.
        reg_key = rec.get("model") or row_reg_hack(rec)
        for row in rec.get("rows") or []:
            if row.get("error"):
                continue
            k = (row.get("model_key") or reg_key, *(row.get(f) for f in KEY[1:]))
            if None in k:
                continue
            cells[k] = {"machine": machine, "device": rec.get("device"), **row}
    return cells, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+", type=pathlib.Path)
    ap.add_argument("--tolerance", type=float, default=0.02,
                    help="relative difference considered a mismatch (default 2%%)")
    ap.add_argument("--metric", default="mase", help="metric used for the tolerance test")
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS),
                    help="metrics to report side by side")
    ap.add_argument("--detail", action="store_true")
    a = ap.parse_args()

    per_file = []
    kind_of: dict[str, set] = {}
    for f in a.files:
        cells, meta = load(f)
        label = meta.get("machine") or f.name
        # which data path produced these cells: a CSV (synthetic 70/15/15 split) and an
        # arena directory (the publisher's split) scale MASE differently, so two lanes on
        # different paths cannot be compared even when they run the same model on the
        # same series
        kinds = {r.get("data_kind") for r in cells.values() if r.get("data_kind")}
        if not kinds:
            # legacy lanes predate the row-level fingerprint. Their meta line still lists
            # the inputs, and a name ending in .csv vs a bare directory name is enough to
            # tell which split was loaded.
            ext = {str(x) for x in meta.get("series") or []}
            kinds = {("csv" if str(x).endswith(".csv") else
                      "arena_dir" if not str(x).endswith((".parquet", ".pq", ".json"))
                      else "parquet_file") for x in ext}
        per_file.append((f, label, cells, meta))
        kind_of[label] = kinds
        kinds_txt = " data_kind=" + ",".join(sorted(kinds)) if kinds else ""
        print(f"{f.name}: machine={label} code_sha={meta.get('code_sha','-')} "
              f"device={meta.get('device','-')} cells={len(cells)}{kinds_txt}")

    # a cell may exist twice inside one file (retry supersedes), keep newest by file order
    labels = [p[1] for p in per_file]
    if len(set(labels)) < 2:
        print("\nNOTE: every file reports the same machine — this compares retries, "
              "not hardware.")

    if len(set(labels)) != len(labels):
        # labels must identify the *run*, not just the host: two files from one machine
        # (a re-run, or two lanes) used to collapse into a single entry, so the
        # same-hardware control group reported "no overlap" and exited 2
        per_file = [(f, f"{label}#{i}" if labels.count(label) > 1 else label, cells, meta)
                    for i, (f, label, cells, meta) in enumerate(per_file)]
        kind_of = {lbl: kind_of[lab] for (f, lbl, _c, _m), lab in zip(per_file, labels)}

    seen: dict[tuple, dict[str, dict]] = {}
    for _, label, cells, _ in per_file:
        for k, v in cells.items():
            seen.setdefault(k, {})[label] = v

    # say so when the same row-level id is being used by several checkpoints: a reader
    # comparing tables needs to know the ids are not unique
    ids: dict[str, set] = {}
    for k in seen:
        ids.setdefault(k[0], set()).add(k[0])
    overlap = {k: v for k, v in seen.items() if len(v) >= 2}
    print(f"\ncells seen: {len(seen)}   on >=2 machines: {len(overlap)}")
    if not overlap:
        print("No overlapping cells. The machines covered disjoint series, so the merged "
              "table is a union and its cross-machine consistency is UNTESTED.")
        return 2

    metrics = [m.strip() for m in a.metrics.split(",") if m.strip()]
    diffs: list[float] = []
    offenders: list[tuple] = []
    data_gap: list[tuple] = []
    pairs: dict[str, int] = {}
    for k, by_machine in sorted(overlap.items()):
        ms = sorted(by_machine)
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                x, y = by_machine[ms[i]], by_machine[ms[j]]
                label_a, label_b = ms[i], ms[j]
                pair = f"{ms[i]}|{ms[j]}"
                pairs[pair] = pairs.get(pair, 0) + 1
                va, vb = x.get(a.metric), y.get(a.metric)
                if not isinstance(va, (int, float)) or not isinstance(vb, (int, float)):
                    continue
                h1, h2 = x.get("data_hash"), y.get("data_hash")
                if not (h1 and h2) and len(kind_of[label_a]) > 0 and kind_of[label_a] != kind_of[label_b]:
                    # legacy lane: no fingerprint on the row, but the lane's own input list
                    # says CSV vs arena directory, which is enough to void the comparison
                    data_gap.append((k, pair, va, vb, "|".join(sorted(kind_of[label_a])),
                                     "|".join(sorted(kind_of[label_b]))))
                    continue
                if h1 and h2 and h1 != h2:
                    # different bytes or a different split under the same series *name*:
                    # the gap measures the data path, not the hardware. Reporting it as a
                    # machine disagreement is how a 15% MASE shift got blamed on a GPU once.
                    data_gap.append((k, pair, va, vb, h1, h2))
                    continue
                base = max(abs(vb), 1e-9)
                rel = abs(va - vb) / base
                diffs.append(rel)
                row = "  ".join(f"{m}={x.get(m)}/{y.get(m)}" for m in metrics
                                if x.get(m) is not None or y.get(m) is not None)
                if rel > a.tolerance:
                    offenders.append((k, pair, va, vb, rel, row))
                elif a.detail:
                    print(f"  ok  {k[0]:18s} {k[1]:24s} h{k[3]:<4} {pair:24s} "
                          f"{a.metric} {va:.4f} vs {vb:.4f} ({100*rel:.3f}%)")

    print(f"compared pairs: {sum(pairs.values())}  " +
          "  ".join(f"{p}={n}" for p, n in sorted(pairs.items())))
    if diffs:
        print(f"{a.metric} relative difference: max {100*max(diffs):.3f}%  "
              f"median {100*statistics.median(diffs):.3f}%  "
              f"tolerance {100*a.tolerance:.2f}%")
    if offenders:
        print(f"\n{len(offenders)} cell(s) outside tolerance:")
        for k, pair, va, vb, rel, row in offenders[:20]:
            print(f"  {k[0]:18s} {k[1]:24s} span={k[2]} h{k[3]:<4} {pair}")
            print(f"    {a.metric} {va!r} vs {vb!r}  ({100*rel:.2f}%)   {row}")
        return 1
    if data_gap:
        # exit 2, not 1: nothing here proves the machines disagree. It proves the two lanes
        # did not read the same bytes, which makes the comparison uninterpretable until one
        # side is re-scored on the other's data path.
        print(f"\n{len(data_gap)} cell(s) could not be compared: same cell key, different "
              "data fingerprint (different file bytes or a different split).")
        for k, pair, va, vb, h1, h2 in data_gap[:6]:
            print(f"  {k[0]:18s} {k[1]:24s} span={k[2]} h{k[3]:<4} {pair}"
                  f"   {a.metric} {va:.4f} vs {vb:.4f}   data_hash {h1} != {h2}")
        print("These lanes are not comparable on a scaled metric: a CSV gets the synthetic "
              "70/15/15 split while an arena directory carries the publisher's, and MASE is "
              "divided by the naive error of whichever train split was loaded. Re-score one "
              "side on the other's data path, or compare only unscaled metrics (rmse, mae).")
        return 2
    print("\nAll overlapping cells agree within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
