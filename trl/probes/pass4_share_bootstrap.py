"""Is the leaderboard ranking real, or is it the window sample talking?

The headline table ranks models by the share of cells where their MASE beats the naive row on the
same (series, span, horizon) cell. With 101 series and 8 windows each, that share has sampling
error: resampling *series* (each series contributes its cells as a block, because two horizons of
one series are not independent) gives a 95% interval, and the pair column answers the question the
table invites - how often does A actually beat B, over resamples of the corpus rather than of one
window draw.

Read the intervals, not the ordering: two models whose intervals overlap are not ranked, they are
tied within what this pass can see.
"""
import argparse
import json
import random
import statistics
from pathlib import Path


def merged_cells(paths):
    """cell key -> row, newest file wins (same rule as the report generator)."""
    cells = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if "meta" in rec:
                continue
            key = rec.get("model")
            for r in rec.get("rows", []):
                k = (r.get("model_key") or key, r["series"], r.get("span", "test"),
                     int(r["horizon"]))
                cells[k] = r
    return cells


def beats(paths):
    cells = merged_cells(paths)
    base = {(s, sp, h): float(r["mase"])
            for (m, s, sp, h), r in cells.items()
            if m == "naive" and isinstance(r.get("mase"), (int, float))}
    flags = {}  # model -> series -> list[0/1]
    for (m, s, sp, h), r in cells.items():
        if m == "naive" or not isinstance(r.get("mase"), (int, float)):
            continue
        b = base.get((s, sp, h))
        if not b or b <= 0:
            continue
        flags.setdefault(m, {}).setdefault(s, []).append(1 if r["mase"] < b else 0)
    return flags


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("lanes", nargs="+")
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    flags = beats(a.lanes)
    rng = random.Random(a.seed)
    point, ci, series_pool = {}, {}, {}
    for m, per_series in flags.items():
        common = sorted(per_series)
        series_pool[m] = common
        flat = [v for s in common for v in per_series[s]]
        point[m] = 100.0 * statistics.fmean(flat)
        draws = []
        for _ in range(a.resamples):
            pick = [per_series[rng.choice(common)] for _ in common]
            draws.append(100.0 * statistics.fmean([v for blk in pick for v in blk]))
        draws.sort()
        ci[m] = (draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))], len(common))

    print(f"cluster bootstrap over series ({a.resamples} resamples, seed {a.seed})\n")
    print("| model | cells | beat naive | 95% CI |")
    print("|---|--:|--:|---|")
    for m in sorted(point, key=lambda k: -point[k]):
        lo, hi, n = ci[m]
        cells = sum(len(v) for v in flags[m].values())
        print(f"| `{m}` | {cells} | {point[m]:.1f}% | {lo:.1f}% .. {hi:.1f}% (n={n} series) |")

    top = sorted(point, key=lambda k: -point[k])
    print("\npairwise: share of resamples where the row model beats the column model\n")
    for i in range(min(5, len(top))):
        row = []
        for j in range(min(5, len(top))):
            if i == j:
                row.append("  - ")
                continue
            A, B = top[i], top[j]
            shared = sorted(set(series_pool[A]) & set(series_pool[B]))
            if not shared:
                row.append(" n/a")
                continue
            wins = 0
            for _ in range(a.resamples):
                pick = [rng.choice(shared) for _ in shared]
                sa = statistics.fmean([v for s in pick for v in flags[A][s]])
                sb = statistics.fmean([v for s in pick for v in flags[B][s]])
                if sa > sb:
                    wins += 1
            row.append(f"{100.0*wins/a.resamples:5.1f}")
        print(f"  {A:22s} " + " ".join(row))
    print(f"\nheader order: {', '.join(top[:5])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
