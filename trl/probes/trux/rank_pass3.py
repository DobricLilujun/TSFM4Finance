#!/usr/bin/env python3
"""Headline table for a merged effectiveness pass: median MASE relative to naive.

Run from the harness root:

    python probes/trux/rank_pass3.py reports/pass3_trux/*.jsonl reports/pass3_watson/*.jsonl

Merge semantics must match `scripts/sweep_evaluate.py --report`: one cell per
(model, series, span, horizon), later files winning. This script keeps its own copy of
that rule on purpose, so the published table can be reproduced even if the report
generator changes.

Three things a naive version of this script gets wrong, all of which happened:

* NaN in a metric column is aggregated as a number. Rows now carry None for "no
  measurement", and anything non-finite is skipped and counted, not averaged in.
* A cell where every window was non-finite has n_windows == 0 and no metrics; it must
  appear in the accounting, not silently vanish from the denominator.
* Models are not all scored at the same context. `min_context` raises TTM to its declared
  length, so the table prints the ctx column instead of implying uniformity.
"""
from __future__ import annotations

import glob
import json
import math
import statistics as st
import sys

fin = lambda v: isinstance(v, (int, float)) and math.isfinite(v)  # noqa: E731


def load(files):
    cells = {}
    for f in sorted(files, key=lambda p: __import__("os").path.getmtime(p)):
        for line in open(f):
            rec = json.loads(line)
            if rec.get("meta") or not rec.get("model"):
                continue
            for x in rec.get("rows", []):
                x["_model"] = rec["model"]
                x["_device"] = rec.get("device") or "?"
                cells[(rec["model"], x["series"], x.get("span", "test"),
                       int(x["horizon"]))] = x
    return list(cells.values())


def main():
    args = sys.argv[1:]
    files = [p for a in args for p in glob.glob(a)] or glob.glob("reports/**/lane_*.jsonl",
                                                                recursive=True)
    rows = load(files)
    key = lambda x: (x["series"], x.get("span", "test"), int(x["horizon"]))  # noqa: E731
    base = {key(x): x.get("mase") for x in rows if x["_model"] == "naive"}

    agg, dropped, empty = {}, 0, 0
    ctxs, devices = {}, {}
    for x in rows:
        m = x["_model"]
        dropped += x.get("pred_nan_windows", 0)
        empty += (x.get("n_windows", 1) == 0)
        ctxs.setdefault(m, set()).add(x.get("ctx"))
        devices.setdefault(m, set()).add(x["_device"])
        if m == "naive":
            continue
        a, b = x.get("mase"), base.get(key(x))
        if not (fin(a) and fin(b)) or b == 0:
            continue
        d = agg.setdefault(m, {"ratio": [], "w": 0, "n": 0, "mase": [], "cov": []})
        d["ratio"].append(a / b)
        d["n"] += 1
        d["w"] += (a < b)
        d["mase"].append(a)
        if fin(x.get("coverage_40")):
            d["cov"].append(x["coverage_40"])

    hdr = "%-19s %5s %10s %16s %10s %8s %-8s %-6s" % (
        "model", "cells", "beat naive", "med MASE/naive", "med MASE", "cov@40", "device", "ctx")
    print(hdr)
    print("-" * len(hdr))
    for m, d in sorted(agg.items(), key=lambda kv: st.median(kv[1]["ratio"])):
        cov = ("%.2f" % st.median(d["cov"])) if d["cov"] else "—"
        print("%-19s %5d %9.0f%% %16.2f %10.2f %8s %-8s %-6s" % (
            m, d["n"], 100.0 * d["w"] / d["n"], st.median(d["ratio"]),
            st.median(d["mase"]), cov, "/".join(sorted(devices[m])),
            "/".join(str(c) for c in sorted(ctxs[m]))))

    models = {x["_model"] for x in rows}
    print()
    print("cells=%d series=%d models=%d" % (len(rows), len({x["series"] for x in rows}),
                                            len(models)))
    print("cells with nothing scored (all windows non-finite): %d" % empty)
    print("windows dropped for a non-finite forecast: %d" % dropped)
    print("models with no finite MASE anywhere: %s" % sorted(models - set(agg)))
    print("models scored at a non-protocol context: %s"
          % sorted(m for m, cs in ctxs.items() if cs != {256}))


if __name__ == "__main__":
    main()
