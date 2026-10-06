"""Merge standard-bench evaluation results into the public leaderboard and
rebuild the GitHub Pages deploy bundle.

Reads docs/standard_bench_report.jsonl (produced by eval_standard_bench.py),
converts each row to the leaderboard.json schema, replaces any existing
`std_*` entries in assets/leaderboard/leaderboard.json, and re-runs
make_deploy.py so assets/outputs/deploy/index.html picks up the new rows.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPORT = ROOT / "docs" / "standard_bench_report.jsonl"
LEADERBOARD = ROOT / "assets" / "leaderboard" / "leaderboard.json"
DEPLOY_SCRIPT = ROOT / "scripts" / "make_deploy.py"

# metrics the page's leaderboard row does not recompute from, but the schema
# expects; set to neutral values so downstream code never sees a missing key.
NEUTRAL = {"mse": 0.0, "directional_accuracy": 0.0, "return_rank_corr": 0.0,
           "pearson": 0.0, "n_points": 0}


def to_leaderboard_row(row: dict) -> dict:
    """Convert an eval_standard_bench.jsonl row into a leaderboard entry."""
    metrics = dict(row.get("metrics", {}))
    for k, v in NEUTRAL.items():
        metrics.setdefault(k, v)
    metrics["score"] = row.get("score") or 0.0
    return {
        "dataset": row["dataset"],
        "model": row["model"],
        "mode": "open",
        "domain": row["domain"],
        "task": "forecast",
        "frequency": row["freq"],
        "score": row.get("score") or 0.0,
        "metrics": metrics,
        "split": row.get("split", "validation"),
        "note": "standard-bench",
    }


def main():
    rows = [json.loads(ln) for ln in REPORT.read_text().splitlines() if ln.strip()]
    good = [r for r in rows if not r.get("error")]
    entries = [to_leaderboard_row(r) for r in good]

    old = json.loads(LEADERBOARD.read_text()) if LEADERBOARD.exists() else []
    # drop any previous std_* entries so we don't duplicate on re-run
    kept = [r for r in old if not str(r.get("dataset", "")).startswith("std_")]
    merged = kept + entries
    LEADERBOARD.write_text(json.dumps(merged, indent=1, ensure_ascii=False), encoding="utf-8")

    n = len(entries)
    print(f"added {n} standard-bench rows ({len(kept)} retained, "
          f"{sum(1 for r in rows if r.get('error'))} errors skipped)")
    print(f"leaderboard now {len(merged)} rows -> {LEADERBOARD}")

    # rebuild the Pages deploy bundle
    import runpy
    print("rebuilding deploy bundle ...")
    runpy.run_path(str(DEPLOY_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
