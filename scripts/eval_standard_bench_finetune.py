"""Scoped Chronos-2 fine-tuning for the standard-bench datasets.

The installed chronos-2.3.x exposes a real training entry point only for
Chronos-2 (Chronos2Pipeline.fit); chronos-bolt-base and timesfm-2.5-200m are
forecast-only in this stack, so full fine-tuning of those two is not supported.
We therefore add a *trained* comparison by fine-tuning Chronos-2 on each
dataset's TRAIN split, then forecasting the VAL/TEST splits through the same
rolling-window framework the zero-shot models use (so metrics are comparable).

Scope: CPU fine-tuning is slow, so this runs on a representative subset
(1 daily series per domain by default) and a large rolling step. Results are
appended to docs/standard_bench_report.jsonl as model="chronos-2-finetuned".
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

from tsfm_eval import load_dataset, list_datasets
from tsfm_eval.evaluation.benchmark import run_benchmark
from tsfm_eval.models.base import BaseModel, PredictionResult
from tsfm_eval.schemas import TaskType

OUT = pathlib.Path(__file__).resolve().parent.parent / "docs"
METRIC_KEYS = ["mae", "smape", "mape", "rmse", "mase"]


class Chronos2FinetunedModel(BaseModel):
    """A per-dataset fine-tuned Chronos-2 (already trained on that series)."""
    supports = {TaskType.FORECAST}

    def __init__(self, pipe, name="chronos-2-finetuned"):
        super().__init__(name)
        self._pipe = pipe

    def predict(self, context_df, meta, horizon=1):
        series = np.array(context_df[meta.target].to_numpy(dtype=float), copy=True)
        inp = torch.from_numpy(series)[None, None, :]  # (1, 1, T) - chronos-2 needs 3-d
        fc = np.asarray(self._pipe.predict(inp, prediction_length=int(horizon)))
        # chronos-2 predict returns (n_series, n_variates, n_quantiles, horizon)
        if fc.ndim == 4:
            point = fc[0, 0, fc.shape[2] // 2, :]     # median quantile
        elif fc.ndim == 3:
            point = fc[0, fc.shape[1] // 2, :]
        elif fc.ndim == 2:
            point = fc[0]
        else:
            point = fc
        return PredictionResult(point=point)


def _scope(standard, args):
    """Pick the datasets to fine-tune: default = 1 daily series per domain."""
    if args.names:
        return [load_dataset(n) for n in args.names]
    if args.domains:
        standard = [m for m in standard if m.domain.value in args.domains]
    picks = {}
    for m in standard:
        if m.frequency.value != "day":            # CPU: daily only for fine-tuning
            continue
        picks.setdefault(m.domain.value, m.name)
    return [load_dataset(n) for n in picks.values()]


def run(args):
    from chronos import Chronos2Pipeline
    standard = [m for m in list_datasets() if m.name.startswith("std_")]
    datasets = _scope(standard, args)
    print(f"fine-tuning {len(datasets)} datasets: {[d.meta.name for d in datasets]}")

    j = pathlib.Path(args.out).with_suffix(".jsonl")
    existing = [json.loads(l) for l in j.read_text(encoding="utf-8").splitlines() if l.strip()] \
        if j.exists() else []
    # drop any prior fine-tuned rows so re-runs don't duplicate
    existing = [r for r in existing if r.get("model") != "chronos-2-finetuned"]
    fresh = []

    for split in ([args.split] if args.split != "both" else ["validation", "test"]):
        for ds in datasets:
            m = ds.meta
            pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2")
            x = ds.train_df[m.target].to_numpy(dtype=float).reshape(1, 1, -1)
            print(f"[{split:10s}] fine-tune {m.name} ({m.domain.value}/{m.frequency.value}) ...",
                  flush=True)
            try:
                pipe.fit(inputs=x, prediction_length=int(m.horizon),
                         context_length=int(m.lookback), learning_rate=1e-5,
                         num_steps=args.steps, batch_size=64, remove_printer_callback=True)
                model = Chronos2FinetunedModel(pipe)
                entries = run_benchmark(model, [ds], split=split, step=args.step)
                e = entries[0] if entries else None
                if e is None:
                    fresh.append({"split": split, "model": "chronos-2-finetuned",
                                  "dataset": m.name, "domain": m.domain.value,
                                  "freq": m.frequency.value, "error": "no entry"})
                    continue
                metrics = {k: v for k, v in e.metrics.items()
                           if k in METRIC_KEYS and v == v and not (isinstance(v, float) and np.isnan(v))}
                fresh.append({
                    "split": split, "model": "chronos-2-finetuned", "dataset": e.dataset,
                    "domain": e.domain.value, "freq": e.frequency.value,
                    "metrics": {k: round(v, 5) for k, v in metrics.items()},
                    "score": round(e.score, 4) if e.score is not None else None,
                })
            except Exception as ex:
                print(f"   ERROR {m.name}: {type(ex).__name__}: {ex}", flush=True)
                fresh.append({"split": split, "model": "chronos-2-finetuned", "dataset": m.name,
                              "domain": m.domain.value, "freq": m.frequency.value,
                              "error": f"{type(ex).__name__}: {str(ex)[:120]}"})
        _flush(existing + fresh, args)
    all_rows = existing + fresh
    _flush(all_rows, args)
    print(f"appended {len(fresh)} fine-tuned rows; total {len(all_rows)} -> {j}")


def _flush(rows, args):
    j = pathlib.Path(args.out).with_suffix(".jsonl")
    with j.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=100, help="rolling-window subsample for the forecast")
    ap.add_argument("--split", default="both", choices=["validation", "test", "both"])
    ap.add_argument("--steps", type=int, default=100, help="chronos-2 fine-tune num_steps")
    ap.add_argument("--domains", nargs="+", default=None)
    ap.add_argument("--names", nargs="+", default=None, help="explicit dataset names to fine-tune")
    ap.add_argument("--out", default=str(OUT / "standard_bench_report.md"))
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
