#!/usr/bin/env python3
"""MOMENT-1 forecasting head: what the released checkpoint can and cannot do.

Established by moment_probe.py, now measured rather than asserted:

* the released checkpoint carries one head tensor — `head.linear` (8×d_model) — which is
  the **reconstruction** head. `from_pretrained(task_name="forecasting")` loads that head
  unchanged, and its output has the *input* length (a 21-step request returns ctx points).
* `model.init()` — which the model card tells you to call — swaps in a forecasting head of
  shape (forecast_horizon, d_model × 64). Two consequences fall out of that single shape:
    - the head's fan-in is `d_model × 64 patches`, i.e. it accepts **exactly 512 input
      points** (MOMENT's `seq_len`); a 256-point context dies in the matmul,
    - `forecast_horizon` is baked into the weights, so one model instance serves one
      horizon; horizons are not a free parameter at inference time.
    and the swapped head is **randomly initialised** (the library warns: "Only
    reconstruction head is pre-trained. Classification and forecasting heads must be
    fine-tuned."), on CPU, so a caller must move the model back to the GPU afterwards.

So a "MOMENT zero-shot forecast" is not a thing: the honest row is *head requires
fine-tuning*, and any number produced by the untrained head is a measurement of a random
linear map, reported here as a reference point rather than as performance.

Runs one size per GPU: small on cuda:0, large on cuda:1.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from momentfm import MOMENTPipeline

DATA = Path("/raid/yewei/projects_yewei/TSFM4Finance/assets/data")
SERIES = {"sp500_daily": DATA / "open" / "sp500_daily",
          "audusd_fx": DATA / "standard" / "std_fx_1d_audusd"}
CTX_VALID = 512          # MOMENT's seq_len: the only context the head accepts
CTX_NEG = 256            # negative control: what a generic harness would feed


def values(d: Path) -> np.ndarray:
    out = []
    for sp in ("train", "validation", "test"):
        df = pd.read_parquet(d / f"{sp}.parquet")
        tgt = "close" if "close" in df.columns else [c for c in df.columns
                                                    if c != "timestamp"][0]
        out.append(df[tgt].to_numpy(dtype=float))
    return np.concatenate(out)


def mase(y, p, hist):
    d = np.abs(np.diff(hist))
    s = d.mean() if len(d) and d.mean() > 0 else 1.0
    return float(np.nanmean(np.abs(y - p)) / s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizons", default="21,63,126")
    a = ap.parse_args()
    rows: list[dict] = []
    for h in [int(x) for x in a.horizons.split(",")]:
        m = MOMENTPipeline.from_pretrained(
            str(a.weights), model_kwargs={"task_name": "forecasting",
                                          "forecast_horizon": h})
        m.init()                       # installs the (untrained) head for this horizon
        m.to(a.device).eval()
        for name, d in SERIES.items():
            y_all = values(d)
            for ctx in (CTX_NEG, CTX_VALID):
                if len(y_all) < ctx + h + 5:
                    continue
                starts = np.linspace(ctx, len(y_all) - h, 6).astype(int)
                ctxs = [y_all[s - ctx:s] for s in starts]
                targs = [y_all[s:s + h] for s in starts]
                x = torch.tensor(np.stack(ctxs)[:, None, :], dtype=torch.float32,
                                 device=a.device)
                mask = torch.ones(x.shape[0], ctx, dtype=torch.long, device=a.device)
                t0 = time.time()
                try:
                    with torch.no_grad():
                        o = m.forecast(x_enc=x, input_mask=mask)
                    fc = o.forecast.detach().cpu().numpy()[:, 0, :]
                except Exception as e:                     # noqa: BLE001
                    rows.append({"series": name, "horizon": h, "ctx": ctx,
                                 "status": "failed",
                                 "error": f"{type(e).__name__}: {str(e)[:150]}"})
                    print(f"  {name} h{h} ctx{ctx}: FAIL {str(e)[:90]}", flush=True)
                    continue
                if fc.shape[1] != h:
                    rows.append({"series": name, "horizon": h, "ctx": ctx,
                                 "status": "shape_mismatch", "output_length": fc.shape[1]})
                    print(f"  {name} h{h} ctx{ctx}: out {fc.shape[1]}", flush=True)
                    continue
                naive = np.stack([np.repeat(c[-1:], h) for c in ctxs])
                r = {"series": name, "horizon": h, "ctx": ctx, "status": "ok",
                     "head_untrained": True,
                     "seconds": round(time.time() - t0, 1),
                     "mase": float(np.mean([mase(t, p, c) for t, p, c in
                                            zip(targs, fc, ctxs)])),
                     "naive_mase": float(np.mean([mase(t, p, c) for t, p, c in
                                                  zip(targs, naive, ctxs)])),
                     "rmse": float(np.mean([np.sqrt(np.mean((t - p) ** 2))
                                            for t, p in zip(targs, fc)]))}
                rows.append(r)
                print("  " + json.dumps(r), flush=True)
        del m
        torch.cuda.empty_cache()
    Path(a.out).write_text(json.dumps({"weights": a.weights, "rows": rows}, indent=1))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
