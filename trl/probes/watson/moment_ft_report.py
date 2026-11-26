"""Read the MOMENT fine-tune lane and print the comparison the matrix cannot show.

The matrix answers "can this family be fine-tuned at all". This answers the next question,
"does fine-tuning help on financial price data", which needs a baseline in the same table.
MOMENT is the only family where we have both zero-shot and fine-tuned cells for the same
series, because its zero-shot forecasting head is randomly initialised (docs/08), so the
zero-shot row is the floor and fine-tuning is the only way it can be anything else.

Usage: python probes/watson/moment_ft_report.py reports/moment_ft_watson/lane_ft.jsonl
"""
from __future__ import annotations

import collections
import json
import pathlib
import statistics
import sys


def main(path: str) -> int:
    recs = [json.loads(l) for l in pathlib.Path(path).read_text().splitlines() if l.strip()]
    runs = [r for r in recs if "rows" in r]

    def sname(r):
        return r["series"].split("/")[-1].replace(".csv", "")

    naive = {(sname(r), x["horizon"]): x["mase"]
             for r in runs if r["model"] == "naive"
             for x in r["rows"] if x.get("mase") is not None}
    by: dict = collections.defaultdict(list)
    flags: dict = {}
    for r in runs:
        s = sname(r)
        tag = (r["model"], r["adaptation"], r.get("lr"), r.get("peft"))
        for x in r["rows"]:
            if x.get("mase") is not None and (s, x["horizon"]) in naive:
                by[(tag, x["horizon"])].append(x["mase"] / naive[(s, x["horizon"])])
                flags[tag] = x.get("meta") or {}

    print("%-15s %-20s %-8s %-11s %3s %3s %10s %6s %6s  flags"
          % ("model", "adaptation", "lr", "peft", "h", "n", "med /naive", "best", "worst"))
    for (tag, h), v in sorted(by.items(), key=lambda kv: (kv[0][0][0], kv[0][0][1],
                                                          str(kv[0][0][2]), kv[0][1])):
        model, adapt, lr, peft = tag
        fl = flags[tag]
        print("%-15s %-20s %-8s %-11s %3d %3d %10.2f %6.2f %6.2f  %s"
              % (model, adapt, str(lr), str(peft), h, len(v),
                 statistics.median(v), min(v), max(v),
                 " ".join(f"{k}={fl[k]}" for k in sorted(fl)) or "-"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1
                          else "reports/moment_ft_watson/lane_ft.jsonl"))
