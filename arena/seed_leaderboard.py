"""Re-seed the leaderboard with REAL evaluations across models x datasets.

For each (model, dataset) we run the arena's rolling-window evaluation on the
dataset's val+test windows and store the averaged composite score. Closed
proprietary datasets are seeded in private mode (score reported, not ranked
publicly) but we still record a placeholder so the table is not empty.

Run:  uv run python -m arena.seed_leaderboard
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from arena.datasets import _read_raw  # noqa: F401 (keeps import graph warm)
from backend.app import _evaluate_dataset, _load_dataset, _to_native

LB_PATH = Path(__file__).resolve().parent.parent / "backend" / "leaderboard.json"

# (model, [datasets to run on])
# Baselines + ARIMA are cheap -> run on a broad set. Chronos/TimesFM are slow ->
# run only on a representative real-data spread so the leaderboard is meaningful.
FAST = ["naive", "moving_average", "seasonal_naive", "random_walk", "arima"]
SLOW = ["chronos", "timesfm"]

FAST_DATASETS = [
    "sp500_daily", "nasdaq_daily", "btc_crypto", "eur_usd_fx",
    "gold_commodity", "us_treasury_7y", "eth_crypto", "russell2000_daily",
]
SLOW_DATASETS = [
    "sp500_daily", "nasdaq_daily", "btc_crypto", "eur_usd_fx",
    "gold_commodity", "us_treasury_7y",
]
# Closed/proprietary: private-mode placeholder (no real data) — still listed.
CLOSED_DATASETS = ["priv_fund_flow", "priv_broker_order",
                   "priv_credit_risk", "priv_crypto_tick"]


def _record(meta, model, metrics, mode="open"):
    return {
        "dataset": meta.name, "model": model, "mode": mode,
        "domain": meta.domain.value, "task": meta.task.value,
        "frequency": meta.frequency.value,
        "score": float(metrics.get("score", 0.0)),
        "metrics": _to_native(metrics),
    }


def main():
    records: list[dict] = []
    # ---- fast models across broad real/synthetic spread ----
    for model in FAST:
        for ds in FAST_DATASETS:
            try:
                meta = _load_dataset(ds)
                metrics = _evaluate_dataset(meta, model, None, None)
                records.append(_record(meta, model, metrics))
                s = metrics.get("score", 0.0)
                print(f"  [fast ] {model:15s} {ds:20s} score={s:.3f}",
                      flush=True)
            except Exception as e:
                print(f"  [fast ] {model:15s} {ds:20s} FAIL {e}", flush=True)

    # ---- slow foundation models on a real spread ----
    for model in SLOW:
        for ds in SLOW_DATASETS:
            try:
                meta = _load_dataset(ds)
                metrics = _evaluate_dataset(meta, model, None, None)
                records.append(_record(meta, model, metrics))
                s = metrics.get("score", 0.0)
                print(f"  [slow ] {model:15s} {ds:20s} score={s:.3f}",
                      flush=True)
            except Exception as e:
                print(f"  [slow ] {model:15s} {ds:20s} FAIL {e}", flush=True)

    # ---- closed/proprietary: private-mode placeholders ----
    for ds in CLOSED_DATASETS:
        try:
            meta = _load_dataset(ds)
            # private mode: we do NOT publish, but keep a record so the API can
            # show that a closed slot exists. Score computed on the synthetic
            # stub only (documented as placeholder, requires user data).
            metrics = _evaluate_dataset(meta, "naive", None, None)
            rec = _record(meta, "naive (placeholder)", metrics, mode="private")
            rec["note"] = "PROPRIETARY placeholder — no real data; " \
                          "requires user data / model upload."
            records.append(rec)
            print(f"  [closed] {ds:20s} (private placeholder)", flush=True)
        except Exception as e:
            print(f"  [closed] {ds:20s} FAIL {e}", flush=True)

    # ---- write leaderboard ----
    LB_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(f"\nSeeded {len(records)} leaderboard records -> {LB_PATH}")


if __name__ == "__main__":
    main()