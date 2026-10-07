"""Validation-set benchmark for the standard-bench datasets.

Runs run_benchmark(..., split="validation") for Chronos-Bolt + TimesFM-2.5-200m
(proxy for timesfm-2.0-500m-pytorch, un-loadable by the installed timesfm 2.3.x)
plus simple baselines, then aggregates by domain / frequency.

run_benchmark returns list[LeaderboardEntry] with a `metrics` dict
(mae/smape/mape/rmse/mse/mase/...) and a composite `score`.
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

# --------------------------------------------------------------------------- #
# Neural models (exact weights)
# ---------------------------------------------------------------------------
class ChronosBoltModel(BaseModel):
    """amazon/chronos-bolt-base (lazy-loaded, reused across datasets)."""
    name = "chronos-bolt-base"
    supports = {TaskType.FORECAST}
    _pipe = None

    @classmethod
    def _pipe_(cls):
        if cls._pipe is None:
            from chronos import ChronosBoltPipeline
            cls._pipe = ChronosBoltPipeline.from_pretrained("amazon/chronos-bolt-base")
        return cls._pipe

    def predict(self, context_df, meta, horizon=1):
        series = np.array(context_df[meta.target].to_numpy(dtype=float), copy=True)
        inp = torch.from_numpy(series)[None, :]  # (1, T)
        fc = np.asarray(self._pipe_().predict(inp, prediction_length=int(horizon)))
        if fc.ndim == 3:            # (n_series, n_quantiles, horizon) -> median quantile
            point = fc[0, fc.shape[1] // 2, :]
        elif fc.ndim == 2:
            point = fc[0]
        else:
            point = fc
        return PredictionResult(point=point)


class TimesFMModel(BaseModel):
    """google/timesfm-2.5-200m-pytorch (proxy for timesfm-2.0-500m-pytorch)."""
    name = "timesfm-2.5-200m-pytorch"
    supports = {TaskType.FORECAST}
    _m = None

    @classmethod
    def _model_(cls):
        if cls._m is None:
            from timesfm import TimesFM_2p5_200M_torch as M, ForecastConfig
            cls._m = M.from_pretrained("google/timesfm-2.5-200m-pytorch", torch_compile=False)
            cls._m.compile(ForecastConfig(max_context=512, max_horizon=128, normalize_inputs=True))
        return cls._m

    def predict(self, context_df, meta, horizon=1):
        series = np.array(context_df[meta.target].to_numpy(dtype=float), copy=True)
        out = np.asarray(self._model_().forecast(horizon=int(horizon), inputs=[series])[0])
        point = np.median(out, axis=0) if out.ndim > 1 else out
        return PredictionResult(point=point)


# --------------------------------------------------------------------------- #
# Baselines (context for the neural models)
# ---------------------------------------------------------------------------
def _last(context_df, meta):
    return context_df[meta.target].to_numpy(dtype=float)


class NaiveModel(BaseModel):
    name = "naive"; supports = {TaskType.FORECAST}
    def predict(self, context_df, meta, horizon=1):
        return PredictionResult(point=np.full(int(horizon), float(_last(context_df, meta)[-1])))


class RandomWalkModel(BaseModel):
    name = "random_walk"; supports = {TaskType.FORECAST}
    def predict(self, context_df, meta, horizon=1):
        return PredictionResult(point=np.full(int(horizon), float(_last(context_df, meta)[-1])))


class MovingAverageModel(BaseModel):
    name = "moving_average"; supports = {TaskType.FORECAST}
    def predict(self, context_df, meta, horizon=1):
        arr = _last(context_df, meta)
        k = 48 if meta.frequency.value == "hour" else 14
        return PredictionResult(point=np.full(int(horizon), float(np.mean(arr[-k:]))))


class SeasonalNaiveModel(BaseModel):
    name = "seasonal_naive"; supports = {TaskType.FORECAST}
    def predict(self, context_df, meta, horizon=1):
        arr = _last(context_df, meta)
        period = 24 if meta.frequency.value == "hour" else 7
        return PredictionResult(point=np.full(int(horizon), float(arr[max(0, len(arr) - period)])))


class ArimaModel(BaseModel):
    """Auto-ARIMA baseline (pmdarima). Falls back to naive on fit failure."""
    name = "arima"; supports = {TaskType.FORECAST}
    def predict(self, context_df, meta, horizon=1):
        arr = _last(context_df, meta)
        try:
            from pmdarima import auto_arima
            m = auto_arima(arr, start_p=0, start_q=0, max_p=2, max_q=2, max_d=2,
                           seasonal=False, suppress_warnings=True, n_jobs=1, maxiter=20)
            # pmdarima 2.x ARIMA exposes .predict(steps) (not .forecast)
            fc = np.asarray(m.predict(int(horizon)), dtype=float)
            if len(fc) >= int(horizon) and np.all(np.isfinite(fc)):
                return PredictionResult(point=fc[:int(horizon)])
        except Exception:
            pass
        return PredictionResult(point=np.full(int(horizon), float(arr[-1])))  # naive fallback


NEURAL = [ChronosBoltModel, TimesFMModel]
BASELINES = [NaiveModel, MovingAverageModel, RandomWalkModel, SeasonalNaiveModel, ArimaModel]


def run(args):
    standard = [m for m in list_datasets() if m.name.startswith("std_")]
    if args.domains:
        standard = [m for m in standard if m.domain.value in args.domains]
    if args.smoke:
        picks = {}
        for m in standard:
            key = (m.domain.value, m.frequency.value)
            picks.setdefault(key, m.name)
        standard = [load_dataset(n) for n in picks.values()]
    else:
        standard = [load_dataset(m.name) for m in standard]
    if args.limit:
        standard = standard[: args.limit]

    classes = NEURAL if args.neural_only else (BASELINES + NEURAL)
    models = [model_cls(model_cls.name) for model_cls in classes]

    splits = [args.split] if args.split != "both" else ["validation", "test"]
    rows = []
    for split in splits:
        for model in models:
            for ds in standard:
                m = ds.meta
                print(f"[{split:10s}] [{model.name:24s}] {m.name} ({m.domain.value}/{m.frequency.value}) ...",
                      flush=True)
                try:
                    entries = run_benchmark(model, [ds], split=split, step=args.step)
                    e = entries[0] if entries else None
                    if e is None:
                        rows.append({"split": split, "model": model.name, "dataset": m.name,
                                     "domain": m.domain.value, "freq": m.frequency.value,
                                     "error": "no entry"})
                        continue
                    metrics = {k: v for k, v in e.metrics.items()
                               if k in METRIC_KEYS and v == v and not (isinstance(v, float) and np.isnan(v))}
                    rows.append({
                        "split": split, "model": model.name, "dataset": e.dataset,
                        "domain": e.domain.value, "freq": e.frequency.value,
                        "metrics": {k: round(v, 5) for k, v in metrics.items()},
                        "score": round(e.score, 4) if e.score is not None else None,
                    })
                except Exception as ex:
                    print(f"   ERROR {m.name}: {type(ex).__name__}: {ex}", flush=True)
                    rows.append({"split": split, "model": model.name, "dataset": m.name,
                                 "domain": m.domain.value, "freq": m.frequency.value,
                                 "error": f"{type(ex).__name__}: {str(ex)[:120]}"})
                _dump(rows, args)
    _write_report(rows, args)
    print("done.")


def _dump(rows, args):
    p = pathlib.Path(args.out).with_suffix(".jsonl")
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _fmt(xs):
    m = _mean(xs)
    return f"{m:.5g}" if xs else "-"


def _write_report(rows, args):
    ok = [r for r in rows if not r.get("error")]
    models = sorted({r["model"] for r in ok})
    splits = sorted({r.get("split", "validation") for r in ok}) or ["validation"]
    split_desc = {"validation": "2019-01-01 .. 2020-12-31", "test": "2021-01-01 .. 2023-12-31"}
    lines = ["# Standard-Bench Evaluation (by granularity)\n",
             "Reported **per granularity (frequency)** — not per dataset, not per model. Each table "
             "aggregates every dataset at one frequency (across all domains). Mean of per-dataset "
             "aggregate metrics over rolling windows (step subsampling = %d). Lower is better for "
             "MAE/SMAPE/MAPE/RMSE/MASE. Splits: %s.\n"
             % (args.step, ", ".join(f"**{s}** ({split_desc.get(s, '')})" for s in splits))]

    def _model_table(cells_filter):
        """One table: rows = models, columns = metrics, aggregated over `cells_filter`."""
        out = ["| model | n | MAE | SMAPE | MAPE | RMSE | MASE |",
               "|---|---:|---:|---:|---:|---:|---:|"]
        for model in models:
            cells = {k: [] for k in METRIC_KEYS}
            n = 0
            for r in cells_filter:
                if r["model"] != model:
                    continue
                n += 1
                for k in cells:
                    if k in r["metrics"]:
                        cells[k].append(r["metrics"][k])
            out.append(f"| {model} | {n} | {_fmt(cells['mae'])} | {_fmt(cells['smape'])} | "
                      f"{_fmt(cells['mape'])} | {_fmt(cells['rmse'])} | {_fmt(cells['mase'])} |")
        out.append("")
        return out

    def _domain_table(cells_filter):
        """Break the same frequency group down by domain (still grouped under the frequency)."""
        out = ["| domain | model | MAE | SMAPE | MASE |",
               "|---|---|---:|---:|---:|"]
        for dom in sorted({r["domain"] for r in cells_filter}):
            for model in models:
                mae, smape, mase, n = [], [], [], 0
                for r in cells_filter:
                    if r["domain"] != dom or r["model"] != model:
                        continue
                    n += 1
                    if "mae" in r["metrics"]: mae.append(r["metrics"]["mae"])
                    if "smape" in r["metrics"]: smape.append(r["metrics"]["smape"])
                    if "mase" in r["metrics"]: mase.append(r["metrics"]["mase"])
                out.append(f"| {dom} | {model} | {_fmt(mae)} | {_fmt(smape)} | {_fmt(mase)} |")
        out.append("")
        return out

    for split in splits:
        sub = [r for r in ok if r.get("split", "validation") == split]
        lines.append(f"## {split.upper()} split ({split_desc.get(split, '')})\n")
        # PRIMARY axis: granularity (frequency). DAY then HOUR.
        for v in sorted({r["freq"] for r in sub}):
            grp = [r for r in sub if r["freq"] == v]
            lines.append(f"### {v.upper()} granularity\n")
            lines += _model_table(grp)
            lines.append("_by domain within this frequency_\n")
            lines += _domain_table(grp)
    lines.append(f"Total runs: {len(rows)}; errors: {sum(1 for r in rows if r.get('error'))} "
                 f"({[r.get('split')+'-'+r['model']+'/'+r['dataset'] for r in rows if r.get('error')][:12]})\n")
    OUT.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"report -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=1, help="subsample rolling windows (1=full)")
    ap.add_argument("--limit", type=int, default=0, help="max datasets")
    ap.add_argument("--domains", nargs="+", default=None)
    ap.add_argument("--neural-only", action="store_true")
    ap.add_argument("--split", default="both", choices=["validation", "test", "both"])
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end check (1 per domain/freq)")
    ap.add_argument("--out", default=str(OUT / "standard_bench_report.md"))
    ap.add_argument("--report-only", action="store_true",
                    help="skip evaluation; rebuild the report from the existing <out>.jsonl")
    args = ap.parse_args()
    if args.report_only:
        j = pathlib.Path(args.out).with_suffix(".jsonl")
        rows = [json.loads(l) for l in j.read_text(encoding="utf-8").splitlines() if l.strip()]
        _write_report(rows, args)
        return
    run(args)


if __name__ == "__main__":
    main()
