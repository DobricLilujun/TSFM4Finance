"""CLI: ``python -m tsfm4finance.bench.trainers --model <key> --series <s> --budget <b>``.

The registry below is deliberately explicit about which families have a trainer
and which are still pending: the sweep plans fine-tuned cells *only* for
checkpoints that exist, so a pending trainer shows up as a missing cell in
reports/PROGRESS.md instead of a silent gap in the paper.
"""
from __future__ import annotations

import os

import argparse
import json
import sys
import traceback
from .common import BUDGETS, REPO, TrainJob

# family -> trainer module (each one implements its own loop, see module docstrings)
REGISTRY = {
    "timesfm25": "tsfm4finance.bench.trainers.timesfm25",
    "chronos": "tsfm4finance.bench.trainers.chronos2",      # Chronos-2 (vendor fit)
    "moirai": "tsfm4finance.bench.trainers.moirai",
    "ttm": "tsfm4finance.bench.trainers.ttm",
    "moment": "tsfm4finance.bench.trainers.moment",
    "toto2": "tsfm4finance.bench.trainers.toto",
    "kronos": "tsfm4finance.bench.trainers.kronos",
}
# trainer-pending: recorded, never silently skipped
PENDING = {
    "timesfm30": "TimesFM 3.0 exposes no public training forward; use the "
                 "vendor notebook or add a head-only loop (see docs/05)",
    "chronos_bolt": "Chronos-Bolt has no fit(); needs a custom loop over "
                    "ChronosBoltForCausal2Seq (quantile head in scaled space)",
    "timemoe": "Time-MoE fine-tune needs their tokeniser utilities "
               "(github.com/Time-MoE/Time-MoE/tree/main/finetune)",
    "tirex": "TiRex is a zero-shot xLSTM model; no public fine-tune recipe",
    "tirex2": "same as TiRex; also blocked by Triton/CUDA kernel compatibility locally",
    "flowstate": "FlowState fine-tune requires the granite-tsfm streaming trainer",
    "baseline": "statistical baselines have nothing to fine-tune",
}


def _import(family: str):
    import importlib
    return importlib.import_module(REGISTRY[family])


def run_one(job: TrainJob, family: str) -> dict:
    if family not in REGISTRY:
        return {"model": job.model_key, "series": job.series, "budget": job.budget,
                "status": "trainer-pending", "reason": PENDING.get(family, "no trainer")}
    mod = _import(family)
    out = mod.train(job)
    out.setdefault("status", "ok")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="", help="model key from bench/config.yaml")
    ap.add_argument("--series", default="", help="comma-separated dataset names")
    ap.add_argument("--budget", default="full", choices=sorted(BUDGETS))
    ap.add_argument("--all-family", default="", help="train every model of a family, e.g. chronos")
    ap.add_argument("--tiers", default="", help="with --all-family: restrict tiers")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=21)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", default="fp32", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="two optimiser steps: proves the loop is wired, not a fine-tune")
    ap.add_argument("--heads-only", action="store_true",
                    help="ablation: freeze the backbone (recorded in metadata)")
    ap.add_argument("--run-training", action="store_true",
                    help="vendor-stack trainers (Kronos) otherwise only print the command")
    ap.add_argument("--force", action="store_true", help="retrain even if a checkpoint exists")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--ignore-env", action="store_true",
                    help="run even if config.yaml points this model at another venv")
    ap.add_argument("--list", action="store_true", help="show trainer status per family and exit")
    ap.add_argument("--weights-root", default=os.environ.get("TSFM_WEIGHTS_ROOT", ""),
                    help="directory holding git-cloned / hf-downloaded models")
    a = ap.parse_args()
    if a.weights_root:
        os.environ["TSFM_WEIGHTS_ROOT"] = a.weights_root

    if a.list:
        for fam in sorted(set(list(REGISTRY) + list(PENDING))):
            print(f"{fam:<14} {'trainer: ' + REGISTRY[fam] if fam in REGISTRY else 'PENDING: ' + PENDING[fam]}")
        print("\nbudgets:", json.dumps(BUDGETS))
        return

    from tsfm4finance.bench.config import load_grid
    grid = load_grid()
    specs = grid.models()
    if a.all_family:
        picks = [k for k, m in specs.items()
                 if m.family == a.all_family and (not a.tiers or m.tier in a.tiers.split(","))]
    elif a.model:
        picks = [a.model]
    else:
        ap.error("--model or --all-family is required (see --list)")

    series = [s.strip() for s in a.series.split(",") if s.strip()]
    if not series:
        ap.error("--series is required (comma-separated dataset names)")

    results = []
    for key in picks:
        spec = specs.get(key)
        if spec is None:
            print(json.dumps({"model": key, "status": "unknown-model"}))
            continue
        if spec.env and a.device == "cuda":
            here = str(REPO / ".venv2" / "bin" / "python")
            want = grid.python_for(spec)
            if want and str(want) != sys.executable and not a._ignore_env:
                print(json.dumps({"model": key, "status": "wrong-env",
                                  "rerun_with": str(want), "family": spec.family}))
                continue
        for s in series:
            job = TrainJob(model_key=key, series=s, budget=a.budget, ctx=a.ctx,
                           horizon=a.horizon, amp=a.amp, bs=a.bs, lr=a.lr,
                           epochs=a.epochs, device=a.device, seed=a.seed,
                           smoke=a.smoke, heads_only=a.heads_only,
                           run_training=a.run_training,
                           out_dir=a.out_dir or str((REPO / "reports" / "ft_ckpts")))
            if job.ckpt_path().exists() and not a.force:
                results.append({"model": key, "series": s, "budget": a.budget,
                                "status": "cached", "checkpoint": str(job.ckpt_path())})
                continue
            try:
                results.append(run_one(job, spec.family))
            except Exception as e:                                     # noqa: BLE001
                results.append({"model": key, "series": s, "budget": a.budget,
                                "status": "failed", "error": f"{type(e).__name__}: {e}"[:400],
                                "trace": traceback.format_exc()[-600:]})
            print(json.dumps(results[-1], default=str), flush=True)
    print(json.dumps({"summary": {r["status"]: sum(1 for x in results if x["status"] == r["status"])
                                  for r in results}}, indent=1))


if __name__ == "__main__":
    main()
