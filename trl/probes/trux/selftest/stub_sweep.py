#!/usr/bin/env python3
"""Contract stub for trux_lane_runner.py: it accepts exactly what sweep_evaluate.py accepts.

The point is not to fake results, it is that argparse rejects anything the real sweep
would reject. A flag renamed since the flag list was written, or series passed
positionally instead of via repeatable `--series`, fails right here instead of at 3 a.m.
inside a 12-hour sweep. Keep this list in sync with `sweep_evaluate.py --help`.

One deliberate deviation: `allow_abbrev=False`. The real CLI inherits argparse's prefix
matching, which means `--windows=3` is *silently accepted* as `--max-windows=3` and a
typo can turn into a different experiment. The stub refuses to inherit that hazard, so a
runner that leans on abbreviations fails here rather than looking fine in production.

    selftest/run.sh        # builds a fake repo around this stub and runs the runner twice
"""
from __future__ import annotations

import argparse
import json
import pathlib

p = argparse.ArgumentParser(allow_abbrev=False)
p.add_argument("--series", action="append", default=[])   # repeatable, as in the real CLI
p.add_argument("--horizons", default="21")
p.add_argument("--max-windows", dest="max_windows", type=int)
p.add_argument("--ctx", type=int)
p.add_argument("--batch", type=int)
p.add_argument("--seed", type=int)
p.add_argument("--spans")
p.add_argument("--models", default="")
p.add_argument("--tier", default="")
p.add_argument("--all", action="store_true")
p.add_argument("--device")
p.add_argument("--envs")
p.add_argument("--allow-core-fallback", action="store_true")
p.add_argument("--timeout", type=int)
p.add_argument("--out", required=True)
p.add_argument("--fresh", action="store_true")
p.add_argument("--report")
p.add_argument("--md")
p.add_argument("--stdout", action="store_true")
a = p.parse_args()

if not a.series:
    raise SystemExit("stub: no --series passed (the runner must use repeatable --series)")
out = pathlib.Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("a") as fh:
    for s in a.series:
        for h in a.horizons.split(","):
            fh.write(json.dumps({"model": "stub", "series": pathlib.Path(s).name,
                                 "horizon": int(h), "device": a.device,
                                 "max_windows": a.max_windows, "seed": a.seed,
                                 "mase": 1.0}) + "\n")
print(f"stub: {len(a.series)} series x {len(a.horizons.split(','))} horizons on {a.device}")
