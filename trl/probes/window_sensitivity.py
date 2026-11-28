"""Does the ranking survive a different number of test windows?

`--max-windows` draws the test-span windows uniformly at random, so the window count is part of the
measurement, not a tuning knob: docs/08 measures that at 8 windows, changing only the seed moves a
series' own MASE ~55% (median) and the paired model/naive ratio ~12%. That makes any published tie
suspect: `chronos-bolt-small` vs `chronos-2` (61.6% of bootstrap resamples) and
`chronos-bolt-base` vs `chronos-t5-small` (53.8%) may be eight-window artifacts.

This probe compares the *same machine, same bytes, same seed, same code* at different window counts.
Cells are grouped by `n_windows`, never merged: a 32-window row and an 8-window row are different
protocols by construction, and the merge machinery would refuse to put them in the same cell anyway.

    probes/window_sensitivity.py reports/pass5_watson_66/lane.jsonl reports/pass7_watson_win/lane_win32.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load(paths):
    """requested window budget -> cell key -> row.

    The group key is what the lane *asked for* (`--max-windows`, in the meta line and stamped on
    rows since pass 6), not the row's `n_windows`: that field counts the windows that survived
    scoring, and a short series caps the sample below the request, so a 64-window lane would split
    into 38/39/40/49/50/64 groups of the same protocol. Lanes predating both fields fall back to
    the scored count and say so in the footer.
    """
    out = defaultdict(dict)
    prov = defaultdict(set)
    for p in paths:
        meta = {}
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if "meta" in rec:
                meta = rec["meta"]
                continue
            for r in rec.get("rows", []):
                scored = int(r.get("n_windows") or 0)
                req = r.get("max_windows") or meta.get("max_windows") or scored
                nw = int(req)
                key = (r.get("model_key") or rec.get("model"), r["series"],
                       r.get("span", "test"), int(r["horizon"]))
                out[nw][key] = r
                prov[nw].add((r.get("machine"), r.get("data_hash"), r.get("code_sha")))
    return out, prov


def _ratios(cells):
    """model -> [mase_model / mase_naive] over the cells where both are finite."""
    naive = {(s, sp, h): r for (m, s, sp, h), r in cells.items() if m == "naive"}
    out = defaultdict(list)
    for (m, s, sp, h), r in cells.items():
        nv = naive.get((s, sp, h))
        if m == "naive" or nv is None:
            continue
        a, b = r.get("mase"), nv.get("mase")
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b > 0:
            out[m].append(a / b)
    return out


def shares(cells):
    """model -> {series -> [1/0 for beat-naive]}, and model -> median per-cell mase ratio."""
    naive = {(s, sp, h): float(r["mase"])
             for (m, s, sp, h), r in cells.items()
             if m == "naive" and isinstance(r.get("mase"), (int, float))}
    flags = defaultdict(lambda: defaultdict(list))
    ratios = defaultdict(list)
    for (m, s, sp, h), r in cells.items():
        if m == "naive" or not isinstance(r.get("mase"), (int, float)):
            continue
        b = naive.get((s, sp, h))
        if not b or b <= 0:
            continue
        flags[m][s].append(1 if r["mase"] < b else 0)
        ratios[m].append(float(r["mase"]) / b)
    return flags, ratios


def win_rate(cells, a, b):
    """Share of (series, span, horizon) cells where a beats b, over the cells both scored."""
    ca = {(s, sp, h): r for (m, s, sp, h), r in cells.items() if m == a}
    cb = {(s, sp, h): r for (m, s, sp, h), r in cells.items() if m == b}
    both = set(ca) & set(cb)
    if not both:
        return None, 0
    wins = sum(1 for k in both
               if isinstance(ca[k].get("mase"), (int, float))
               and isinstance(cb[k].get("mase"), (int, float))
               and ca[k]["mase"] < cb[k]["mase"])
    return wins / len(both), len(both)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lanes", nargs="+")
    ap.add_argument("--pairs", default="chronos-bolt-small>chronos-2,chronos-bolt-base>chronos-t5-small,"
                                      "arima>chronos-bolt-small,chronos-2>chronos-t5-small",
                    help="a>b pairs whose head-to-head rate is printed per window count")
    a = ap.parse_args()

    by_nw, prov = load(a.lanes)
    print("requested window budgets present (machine, data_hash, code_sha sets per group):")
    for nw in sorted(by_nw):
        print(f"  {nw:3d} windows: {len(by_nw[nw]):5d} cells  stacks={sorted(prov[nw])[:3]}")

    models = sorted({m for cells in by_nw.values() for (m, *_r) in cells})
    print("\nbeat-naive share per window count (share of that model's cells beating the naive row "
          "of the same cell):")
    hdr = f"{'model':22s}" + "".join(f"{nw:>12d}w" for nw in sorted(by_nw)) + "     spread"
    print(hdr)
    flags_by_nw = {nw: shares(by_nw[nw])[0] for nw in by_nw}
    for m in models:
        row = []
        for nw in sorted(by_nw):
            per = flags_by_nw[nw].get(m)
            if not per:
                row.append(None)
                continue
            v = [x for s in per for x in per[s]]
            row.append(sum(v) / len(v))
        vals = [x for x in row if x is not None]
        cells = ""
        for x in row:
            cells += f"{'-':>17s}" if x is None else f"{x*100:11.1f}%  "
        spread = f"{(max(vals)-min(vals))*100:6.1f} pp" if len(vals) > 1 else "     -"
        print(f"{m:22s}{cells}{spread}")

    print("\nper-cell mase ratio against the naive row of the same cell, per window count.")
    print("This is where the story is: the 8-window distributions are wider *and* centred closer")
    print("to parity, so a threshold at 1.0 counts more wins than the tighter 32-window estimate")
    print("supports. `within 10%` is the share of cells that cannot tell the two apart.")
    print(f"{'model':22s}" + "".join(f"{nw:>34d}w" for nw in sorted(by_nw)))
    for m in models:
        if m == "naive":
            continue
        line = f"{m:22s}"
        for nw in sorted(by_nw):
            rs = _ratios(by_nw[nw]).get(m)
            if not rs:
                line += f"{'-':>35s}"
                continue
            med = statistics.median(rs)
            # quantiles(rs, n=4) returns the 25/50/75% cut points: the IQR is q[0]..q[2]
            q = statistics.quantiles(rs, n=4) if len(rs) > 3 else [med, med, med]
            near = sum(1 for x in rs if 0.9 <= x <= 1.1) / len(rs)
            line += f"   med {med:5.3f} [{q[0]:5.3f},{q[2]:5.3f}] {near * 100:5.1f}%"
        print(line)

    print("\nhead-to-head, per window count (share of cells the row model beats the column model in):")
    for spec in a.pairs.split(","):
        x, y = (s.strip() for s in spec.split(">"))
        line = f"  {x} > {y}:"
        for nw in sorted(by_nw):
            rate, n = win_rate(by_nw[nw], x, y)
            line += "            -" if rate is None else f"  {rate*100:5.1f}% (n={n})"
        print(line)

    nws = [n for n in sorted(by_nw) if n > 0]     # 0 means "n_windows was never recorded"
    if 0 in by_nw:
        print(f"\nnote: {len(by_nw[0])} cells predate the n_windows field; excluded from the "
              "window comparison (they are still counted in the merged report as their own protocol)")
    print(f"\ncell-level outcome flips (same model+series+horizon, {nws[0] if nws else '?'}w vs "
          f"{nws[-1] if nws else '?'}w):")
    if len(nws) >= 2:
        lo, hi = nws[0], nws[-1]
        flips = defaultdict(lambda: [0, 0])
        for key in set(by_nw[lo]) & set(by_nw[hi]):
            m = key[0]
            if m == "naive":
                continue
            fl = outcome_flip(by_nw[lo], by_nw[hi], key)
            if fl is None:
                continue
            flips[m][1] += 1
            flips[m][0] += fl
        for m in sorted(flips, key=lambda k: -flips[k][0] / max(1, flips[k][1])):
            f, n = flips[m]
            print(f"  {m:22s} {f:3d}/{n:3d} cells change winner  ({100*f/n:4.1f}%)")
        tot = [sum(v[0] for v in flips.values()), sum(v[1] for v in flips.values())]
        print(f"  {'ALL':22s} {tot[0]:3d}/{tot[1]:3d}                     ({100*tot[0]/max(1,tot[1]):4.1f}%)")
    print("\nA cell whose winner flips between 8 and 32 windows was never a measurement of model "
          "quality on that series; it was a measurement of that window draw.")
    return 0


def outcome_flip(lo_cells, hi_cells, key):
    """1 if 'beats naive' differs between the two window counts, None if either side lacks it."""
    a = _beat(lo_cells, key)
    b = _beat(hi_cells, key)
    if a is None or b is None:
        return None
    return int(a != b)


def _beat(cells, key):
    m, s, sp, h = key
    r = cells.get(key)
    nv = cells.get(("naive", s, sp, h))
    if not r or not isinstance(r.get("mase"), (int, float)):
        return None
    if not nv or not isinstance(nv.get("mase"), (int, float)) or nv["mase"] <= 0:
        return None
    return r["mase"] < nv["mase"]


if __name__ == "__main__":
    raise SystemExit(main())
