#!/usr/bin/env python3
"""Does training MOMENT's task head make it competitive on financial series?

MOMENT-1's released checkpoint carries one trained head: `head.linear`, the reconstruction
head. The forecasting head is created by `model.init()` and is randomly initialised, which
pass 3 showed (MASE ~12, 1% of cells beating naive, every row flagged
`head_from_checkpoint: false`). The compatibility matrix can only say the call returned; the
honest entry for MOMENT forecasting is *after head training*, which is what this measures.

Protocol deliberately mirrors the effectiveness sweep so the two are comparable cell by cell:
same context, same stride, same window count, same seed, evaluation always on the `test`
span. Fine-tuning samples windows from the `train` span only (the harness enforces the split,
and the saved checkpoint records `split_end` so a wrongly paired checkpoint is refused).

stdout is a data channel: one JSON line per (model, series, adaptation, budget) at the end;
progress goes to stderr.

    PYTHONPATH=. <moment-env>/bin/python probes/moment_head_finetune.py \
        --series a.csv b.csv --models moment-1-small,moment-1-large \
        --configs heads_only:fs16,heads_only:fs64,full:fs64 \
        --horizons 21,63 --ctx 256 --max-windows 8 --device cuda \
        --out reports/moment_ft/lane_ft.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from tsfm_trl import ForecastTrainer, ModelHub, TrainingConfig
from tsfm_trl.data.series import SeriesDataset
from tsfm_trl.eval.evaluate import forecast_eval
from tsfm_trl.training.checkpoints import describe


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def jnum(v):
    """JSON has no NaN: a loss that never became finite must read as null, not as the bare
    token json.dumps emits (which strict parsers reject)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def evaluate(model, ds, *, horizons, ctx, max_windows, seed, batch):
    """Same call the sweep makes, so a fine-tuned row and a zero-shot row differ only in
    the weights behind them."""
    return forecast_eval(model, ds, horizons=horizons, spans=("test",), ctx=ctx,
                         max_windows=max_windows, batch=batch, seed=seed)


def base_row(rec_model: str, series: str, adaptation: str, cfg) -> dict:
    return {"model": rec_model, "series": series, "adaptation": adaptation,
            "budget": getattr(cfg, "budget", None), "peft": getattr(cfg, "peft", None),
            "ctx": cfg.ctx, "max_windows": cfg.max_windows, "seed": cfg.seed,
            "split_end": None, "protocol": "docs/09_benchmark.md"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", action="append", required=True, help="CSV/parquet series")
    ap.add_argument("--models", default="moment-1-small,moment-1-large")
    ap.add_argument("--configs", default="heads_only:fs16,heads_only:fs64,full:fs64",
                    help="comma list of peft:budget[:lr] — the lr field overrides the budget "
                         "lr; a fine-tune whose loss *rises* is a broken optimisation, not a "
                         "negative result, and lr is the first knob to prove that out")
    ap.add_argument("--horizons", default="21,63")
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--max-windows", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lr", type=float, default=None, help="override the budget lr")
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline-models", default="naive,seasonal_naive,arima",
                    help="scored zero-shot with no fit, so the fine-tune comparison has a "
                         "local reference in the same file (pass-3 naive rows live in a "
                         "different tree and cover a different series split)")
    ap.add_argument("--out-dir", default="reports/moment_ft/ckpts")
    a = ap.parse_args()

    series = list(dict.fromkeys(a.series))
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    def split_cfg(spec: str):
        parts = [x.strip() for x in spec.split(":")]
        peft, budget = parts[0], parts[1]
        lr = float(parts[2]) if len(parts) > 2 and parts[2] else None
        return peft, budget, lr

    configs = [split_cfg(c) for c in a.configs.split(",") if c.strip()]
    hz = tuple(int(h) for h in a.horizons.split(","))

    out = open(a.out, "w")
    t_start = time.time()

    # references first: they cost nothing and every fine-tune delta is meaningless without
    # the same-series naive/ARIMA row measured under the identical protocol
    for mk in [m.strip() for m in a.baseline_models.split(",") if m.strip()]:
        for path in series:
            ds = SeriesDataset.load(path)
            r = dict(base_row(mk, ds.name, "zero-shot", argparse.Namespace(
                ctx=a.ctx, max_windows=a.max_windows, seed=a.seed, budget=None, peft=None)))
            try:
                m = ModelHub.load(mk, device=a.device, batch=a.batch)
                r["rows"] = evaluate(m, ds, horizons=hz, ctx=a.ctx, max_windows=a.max_windows,
                                     seed=a.seed, batch=a.batch)
                r["status"] = "ok"
            except Exception as e:  # noqa: BLE001
                r["status"], r["rows"] = "failed", []
                r["error"] = f"{type(e).__name__}: {e}"[:300]
            out.write(json.dumps(r) + "\n")
            out.flush()
        log(f"== baseline {mk} done")

    for mk in models:
        log(f"== {mk}")
        for path in series:
            ds = SeriesDataset.load(path)
            name = ds.name
            rec = dict(base_row(mk, name, "zero-shot", argparse.Namespace(
                ctx=a.ctx, max_windows=a.max_windows, seed=a.seed, budget=None, peft=None)))
            try:
                # fresh instance per series so nothing leaks between measurements
                m = ModelHub.load(mk, device=a.device, batch=a.batch)
                zs = evaluate(m, ds, horizons=hz, ctx=a.ctx, max_windows=a.max_windows,
                              seed=a.seed, batch=a.batch)
                rec["rows"] = zs
                rec["status"] = "ok"
                out.write(json.dumps(rec) + "\n")
                out.flush()
                m.close() if hasattr(m, "close") else None
            except Exception as e:  # noqa: BLE001 - a row that fails must still be recorded
                out.write(json.dumps({**rec, "status": "failed", "error": f"{type(e).__name__}: {e}"[:400], "rows": []}) + "\n")
                out.flush()
                log(f"  {name} zero-shot FAILED {type(e).__name__}: {str(e)[:90]}")
                continue

            for peft, budget, lr_ov in configs:
                lr = lr_ov if lr_ov is not None else a.lr
                r = dict(base_row(mk, name, f"finetuned:{peft}", argparse.Namespace(
                    ctx=a.ctx, max_windows=a.max_windows, seed=a.seed, budget=budget,
                    peft=peft)))
                r["lr"] = lr
                t0 = time.time()
                try:
                    m2 = ModelHub.load(mk, device=a.device, batch=a.batch)
                    # the horizon drives the head length, so one fit per horizon
                    fits = []
                    for h in hz:
                        cfg = TrainingConfig(ctx=a.ctx, horizon=h, budget=budget, bs=a.batch,
                                             peft=peft, seed=a.seed, lr=lr,
                                             out_dir=a.out_dir,
                                             tag=f"{mk}_{name}_{peft}_{budget}_lr{lr}_h{h}")
                        res = ForecastTrainer(m2, cfg).fit(ds)
                        fits.append({"horizon": h, "status": res.status, "steps": res.steps,
                                     "n_windows": res.n_windows,
                                     "trainable": res.trainable_parameters,
                                     "first_loss": jnum(res.first_loss),
                                     "final_loss": jnum(res.final_loss),
                                     "seconds": round(res.seconds, 2),
                                     "checkpoint": str(res.checkpoint) if res.checkpoint else None,
                                     "trained_parameters": list(res.trained_parameters)[:8],
                                     "fallbacks": res.extra.get("fallbacks", []),
                                     "reason": (res.reason or "")[:300]})
                    r["fit"] = fits
                    r["fit_seconds"] = round(time.time() - t0, 2)
                    ev = evaluate(m2, ds, horizons=hz, ctx=a.ctx, max_windows=a.max_windows,
                                  seed=a.seed, batch=a.batch)
                    r["rows"] = ev
                    r["status"] = "ok" if all(f["status"] == "ok" for f in fits) else "partial"
                    if fits and fits[0]["checkpoint"]:
                        try:
                            r["ckpt"] = {k: v for k, v in describe(fits[0]["checkpoint"]).items()
                                         if k in ("base", "peft", "trainable", "split_end",
                                                  "config", "steps")}
                        except Exception:  # noqa: BLE001 - provenance detail, not a result
                            pass
                    out.write(json.dumps(r) + "\n")
                    out.flush()
                    log(f"  {name} {peft}/{budget}: " +
                        " ".join(f"h{f['horizon']} {f['status']} steps={f['steps']} "
                                 f"loss {f['first_loss']}->{f['final_loss']}" for f in fits))
                except Exception as e:  # noqa: BLE001
                    out.write(json.dumps({**r, "status": "failed", "rows": [],
                                          "error": f"{type(e).__name__}: {e}"[:400]}) + "\n")
                    out.flush()
                    log(f"  {name} {peft}/{budget} FAILED {type(e).__name__}: {str(e)[:90]}")

    out.close()
    log(f"done in {time.time() - t_start:.0f}s -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
