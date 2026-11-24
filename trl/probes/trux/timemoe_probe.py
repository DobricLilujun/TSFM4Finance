#!/usr/bin/env python3
"""TimeMoE-50M capability probe on trux GPUs, ahead of the harness landing here.

Why a standalone probe: `tsfm-trl` is on another machine and trux has no route to it
(laptop, DGX and watson are all unreachable from trux, and the repo had no remote), but
TimeMoE was exactly the row that OOMed on the 6 GB laptop and whose family env was
missing elsewhere. trux has 2x RTX 6000 Ada, so "does TimeMoE-50M run on the real
datasets, and what does it produce" is answerable now.

What this probe does *not* do: reproduce the harness metric path. The MASE/RMSE here are
local implementations, so these rows are comparable to each other, not to
`EFFECTIVENESS.md`. What it does establish is the toolchain (torch/CUDA/transformers +
the remote-code checkpoint), the shape of the model's output, and the input-scale
question — the released 50M head is **deterministic** (`next_tokens = next_tokens_scores`
in `ts_generation_mixin.py`: no sample axis), so any interval it appears to offer comes
from the caller, not the model. Both findings belong in the matrix.

Usage:  /raid/yewei/envs/tsfm-trl/timemoe/bin/python timemoe_probe.py [--quick]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, GenerationConfig

DATA = Path("/raid/yewei/projects_yewei/TSFM4Finance/assets/data")
SERIES = {
    "sp500_daily": DATA / "open" / "sp500_daily",
    "audusd_fx": DATA / "standard" / "std_fx_1d_audusd",
    "btc_crypto": DATA / "open" / "btc_crypto",
    "de_10y_yield": DATA / "standard" / "std_fi_1d_de_10y",
}
CTX, NWIN = 256, 8
HORIZONS = (21, 63, 126)


def series_values(d: Path) -> np.ndarray:
    cols = []
    for split in ("train", "validation", "test"):
        df = pd.read_parquet(d / f"{split}.parquet")
        tgt = "close" if "close" in df.columns else [c for c in df.columns
                                                    if c != "timestamp"][0]
        cols.append(df[tgt].to_numpy(dtype=float))
    return np.concatenate(cols)


def mase(y, p, hist):
    d = np.abs(np.diff(hist))
    scale = d.mean() if len(d) and d.mean() > 0 else 1.0
    return float(np.nanmean(np.abs(y - p)) / scale)


def run(model, ctxs, h, scale: bool):
    """One batched deterministic rollout. `scale` z-normalises each context (and maps
    the forecast back), which is the input-scale question this probe is also answering."""
    stats = [(float(np.mean(c)), float(np.std(c)) or 1.0) for c in ctxs]
    fed = [[(c - m) / s for c, (m, s) in zip(ctxs, stats)]] if scale else [ctxs]
    # the remote code's greedy loop wants a [B, T] float tensor; `inputs=` as a list of
    # lists is the transformers-4.40 idiom and dies in 4.41+ anyway
    t = torch.tensor(np.stack(fed[0]), dtype=torch.float32, device="cuda")
    with torch.no_grad():
        out = model.generate(inputs=t, max_new_tokens=h, do_sample=False,
                             return_dict_in_generate=True)
    seq = out.sequences.detach().cpu().numpy().astype(float)   # (B, ctx + h)
    fc = seq[:, -h:]
    if scale:
        fc = np.stack([f * s + m for f, (m, s) in zip(fc, stats)])
    return fc


def main() -> int:
    quick = "--quick" in sys.argv
    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} "
          f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}",
          flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "Maple728/TimeMoE-50M", device_map="cuda:0", torch_dtype=torch.float32,
        trust_remote_code=True)
    model.eval()
    print(f"loaded {sum(p.numel() for p in model.parameters()):,} params in "
          f"{time.time() - t0:.1f}s", flush=True)

    rows = []
    for name, d in SERIES.items():
        if not d.exists():
            print(f"  {name}: missing {d}")
            continue
        y_all = series_values(d)
        for h in ((21,) if quick else HORIZONS):
            if len(y_all) < CTX + h + 10:
                print(f"  {name} h={h}: span {len(y_all)} too short")
                continue
            starts = np.linspace(CTX, len(y_all) - h, NWIN).astype(int)
            ctxs = [y_all[s - CTX:s] for s in starts]
            targs = [y_all[s:s + h] for s in starts]
            for scale in (False, True):
                t1 = time.time()
                try:
                    fc = run(model, ctxs, h, scale)
                except Exception as e:                     # noqa: BLE001
                    print("  FAIL", name, h, "scale" if scale else "raw",
                          f"{type(e).__name__}: {str(e)[:110]}", flush=True)
                    continue
                naive = np.stack([np.repeat(c[-1:], h) for c in ctxs])
                r = {"series": name, "horizon": h, "input_scale":
                     "znorm" if scale else "raw", "windows": len(starts),
                     "ctx": CTX, "seconds": round(time.time() - t1, 1),
                     "mase": float(np.mean([mase(t, p, c) for t, p, c in
                                            zip(targs, fc, ctxs)])),
                     "naive_mase": float(np.mean([mase(t, p, c) for t, p, c in
                                                  zip(targs, naive, ctxs)])),
                     "rmse": float(np.mean([np.sqrt(np.mean((t - p) ** 2))
                                            for t, p in zip(targs, fc)])),
                     "dir_acc": float(np.mean([
                         np.mean(np.sign(t - c[-1]) == np.sign(p - c[-1]))
                         for t, p, c in zip(targs, fc, ctxs)])),
                     "finite": bool(np.isfinite(fc).all())}
                rows.append(r)
                print("  " + json.dumps(r), flush=True)
            if not quick:
                # determinism check: the released head has no sampling axis
                a = run(model, ctxs[:2], 21, False)
                b = run(model, ctxs[:2], 21, False)
                print("  determinism: max|Δ| over two identical rollouts = "
                      f"{np.max(np.abs(a - b)):.3e}", flush=True)
    out = Path("/tmp/timemoe_probe.json")
    out.write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} probe rows in {time.time() - t0:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
