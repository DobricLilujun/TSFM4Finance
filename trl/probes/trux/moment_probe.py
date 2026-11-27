#!/usr/bin/env python3
"""MOMENT-1 capability probe on trux (2x RTX 6000 Ada), weights cloned by git+LFS.

Why: the registry pointed at `Momonlp/MOMENT-1-*`, which 401s — the real org is
`AutonLab`, so MOMENT was never gated, the id was wrong. Cloning over git (as asked)
also keeps the exact commit that produced the weights:

    git clone https://huggingface.co/AutonLab/MOMENT-1-small
    git clone https://huggingface.co/AutonLab/MOMENT-1-large

What the probe settles (each is a matrix row, not an opinion):

1. MOMENT's task head is chosen at load time by `task_name`, and the released
   checkpoint contains **one** head tensor: `head.linear` of shape (8, 512) — the
   reconstruction head. transformers reports "0 missing keys" for every task because
   the other heads are new modules with different shapes; the library itself warns
   "Only reconstruction head is pre-trained. Classification and forecasting heads must
   be fine-tuned." So a *zero-shot* MOMENT forecast measures a randomly initialised
   linear map, and `model.init()` — which the README tells you to call — re-randomises
   it *after* loading. An adapter that calls `init()` after loading a fine-tuned
   checkpoint silently destroys the trained head.
2. MOMENT ships a `normalizer` (RevIN-style), so unlike TimeMoE the input scale should
   not matter; the probe measures it rather than trusting the module's name.
3. Which capabilities are legitimate: embedding and reconstruction/anomaly use
   pretrained weights; forecasting/classification do not.

Local metrics (MASE/RMSE), same caveat as the timemoe probe: comparable within this
probe, not against EFFECTIVENESS.md.

Usage: CUDA_VISIBLE_DEVICES=0 ... moment_probe.py --weights ... --device cuda --out f.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open

DATA = Path("/raid/yewei/projects_yewei/TSFM4Finance/assets/data")
SERIES = {
    "sp500_daily": DATA / "open" / "sp500_daily",
    "audusd_fx": DATA / "standard" / "std_fx_1d_audusd",
    "btc_crypto": DATA / "open" / "btc_crypto",
    "de_10y_yield": DATA / "standard" / "std_fi_1d_de_10y",
}
CTX, NWIN, HORIZONS = 256, 8, (21, 63, 126)


def values(d: Path) -> np.ndarray:
    out = []
    for split in ("train", "validation", "test"):
        df = pd.read_parquet(d / f"{split}.parquet")
        tgt = "close" if "close" in df.columns else [c for c in df.columns
                                                     if c != "timestamp"][0]
        out.append(df[tgt].to_numpy(dtype=float))
    return np.concatenate(out)


def mase(y, p, hist):
    d = np.abs(np.diff(hist))
    s = d.mean() if len(d) and d.mean() > 0 else 1.0
    return float(np.nanmean(np.abs(y - p)) / s)


def head_audit(weights: Path, device: str) -> list[dict]:
    """Which heads exist in the checkpoint vs which the task requests."""
    from momentfm import MOMENTPipeline
    with safe_open(str(weights / "model.safetensors"), "pt") as f:
        ck = {k: list(f.get_slice(k).get_shape()) for k in f.keys()
              if k.startswith("head")}
    out = []
    cases = (({"task_name": "forecasting", "forecast_horizon": 21}, "forecasting"),
             ({"task_name": "classification", "n_channels": 1, "num_class": 2},
              "classification"),
             ({"task_name": "embedding"}, "embedding"),
             ({"task_name": "reconstruction"}, "reconstruction"))
    for kw, tag in cases:
        # audit on CPU: each instance also materialises its flan-t5 backbone, and four
        # of them on one 48 GB card is what OOMs, not the model under test
        m = MOMENTPipeline.from_pretrained(str(weights), model_kwargs=dict(kw))
        sd = m.state_dict()
        loaded = {k: list(sd[k].shape) for k in sd if k.startswith("head")}
        before = {k: sd[k].clone() for k in loaded}
        m.init()                      # what the README tells you to call
        after = m.state_dict()
        post = {k: list(after[k].shape) for k in after if k.startswith("head")}
        # the head module is *swapped* by init(), not merely re-initialised: from_pretrained
        # installs the checkpoint's reconstruction head, init() replaces it with the task
        # head and randomises it
        swapped = [k for k in post
                   if k not in before or list(after[k].shape) != list(before[k].shape)
                   or not torch.equal(before[k], after[k].clone())]
        out.append({"task_name": tag,
                    "head_after_from_pretrained": loaded,
                    "head_after_init": post,
                    "loaded_head_is_in_checkpoint": {k: (ck[k] if k in ck else None)
                                                     for k in loaded},
                    "init_swapped_or_reinitialised": sorted(swapped)})
        del m, sd, before, after
        torch.cuda.empty_cache()
    return out, ck


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    w = Path(a.weights)
    import momentfm  # noqa: F401 - side-effect: registers the model classes
    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} "
          f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}",
          flush=True)

    audit, ck_head = head_audit(w, a.device)
    print("checkpoint head tensors:", json.dumps(ck_head), flush=True)
    for r in audit:
        print("  audit:", json.dumps(
            {k: (v if k not in ("requested_head",) else
                 {kk: vv for kk, vv in v.items()})
             for k, v in r.items()}), flush=True)

    from momentfm import MOMENTPipeline
    rows = []
    h = 21 if a.quick else 63
    model = MOMENTPipeline.from_pretrained(
        str(w), model_kwargs={"task_name": "forecasting", "forecast_horizon": h})
    model.to(a.device).eval()
    nparam = sum(p.numel() for p in model.parameters())
    # two states, because the README and the checkpoint disagree about what a "loaded
    # model" is: from_pretrained installs the checkpoint's reconstruction head; init()
    # swaps in the (untrained) forecasting head
    head_states = [("head=as-loaded", None), ("head=after-init()", True)]
    print(f"forecasting pipeline loaded, params={nparam:,}", flush=True)

    for state_tag, _ in head_states:
        if state_tag == "head=after-init()":
            model.init()
            # init() rebuilds the head module, and the fresh module is on CPU: calling
            # init() after .to(cuda) leaves the model split across devices unless the
            # caller moves it back — an easy way to "confirm" a model is broken
            model.to(a.device)
        for name, d in SERIES.items():
            if not d.exists():
                continue
            y_all = values(d)
            for hz in ((21,) if a.quick else HORIZONS):
                if len(y_all) < CTX + hz + 10:
                    continue
                starts = np.linspace(CTX, len(y_all) - hz, NWIN).astype(int)
                ctxs = [y_all[s - CTX:s] for s in starts]
                targs = [y_all[s:s + hz] for s in starts]
                for scale in (False, True):
                    mu = np.array([c.mean() for c in ctxs])[:, None]
                    sd = np.array([c.std() or 1.0 for c in ctxs])[:, None]
                    fed = (np.stack(ctxs) - mu) / sd if scale else np.stack(ctxs)
                    x = torch.tensor(fed, dtype=torch.float32, device=a.device)[:, None, :]
                    mask = torch.ones_like(x[:, 0, :], dtype=torch.long)
                    t1 = time.time()
                    try:
                        with torch.no_grad():
                            o = model.forecast(x_enc=x, input_mask=mask)
                        fc = o.forecast.detach().cpu().numpy()[:, 0, :]
                    except Exception as e:                         # noqa: BLE001
                        # a failure is a result: persist the reason, do not just print
                        # it and leave the table with an unexplained hole
                        rows.append({"series": name, "horizon": hz,
                                     "head_state": state_tag,
                                     "input_scale": "znorm" if scale else "raw",
                                     "status": "failed",
                                     "error": f"{type(e).__name__}: {str(e)[:140]}"})
                        print("  FAIL", name, hz, f"{type(e).__name__}: {str(e)[:110]}",
                              flush=True)
                        continue
                    # the as-loaded (reconstruction) head answers with the *input*
                    # length: a 21-step request comes back as ctx points. Record that
                    # instead of scoring it — silently broadcasting a wrong length is
                    # how a bogus metric gets published.
                    if fc.shape[1] != hz:
                        rows.append({"series": name, "horizon": hz,
                                     "head_state": state_tag,
                                     "input_scale": "znorm" if scale else "raw",
                                     "output_length": int(fc.shape[1]),
                                     "shape_mismatch": True,
                                     "illegal_output": bool(
                                         getattr(o, "illegal_output", False))})
                        print("  MISMATCH", name, "h", hz, "-> out", fc.shape[1],
                              flush=True)
                        continue
                    if scale:
                        fc = fc * sd + mu
                    naive = np.stack([np.repeat(c[-1:], hz) for c in ctxs])
                    r = {"series": name, "horizon": hz, "head_state": state_tag,
                         "input_scale": "znorm" if scale else "raw",
                         "windows": len(starts), "ctx": CTX,
                         "seconds": round(time.time() - t1, 1),
                         "mase": float(np.mean([mase(t, p, c) for t, p, c in
                                                zip(targs, fc, ctxs)])),
                         "naive_mase": float(np.mean([mase(t, p, c) for t, p, c in
                                                      zip(targs, naive, ctxs)])),
                         "rmse": float(np.mean([np.sqrt(np.mean((t - p) ** 2))
                                                for t, p in zip(targs, fc)])),
                         "finite": bool(np.isfinite(fc).all()),
                     "illegal_output": bool(getattr(o, "illegal_output", False)),
                     "head_untrained": state_tag == "head=after-init()"}
                    rows.append(r)
                    print("  " + json.dumps(r), flush=True)
    del model
    torch.cuda.empty_cache()

    for task, kwargs, call in (("embedding", {"task_name": "embedding"}, "embed"),
                               ("reconstruction", {"task_name": "reconstruction"},
                                "detect_anomalies")):
        m2 = MOMENTPipeline.from_pretrained(str(w), model_kwargs=dict(kwargs))
        m2.to(a.device).eval()
        y_all = values(SERIES["sp500_daily"])
        x = torch.tensor(y_all[:CTX][None, None, :], dtype=torch.float32,
                         device=a.device)
        mask = torch.ones(1, CTX, dtype=torch.long, device=a.device)
        with torch.no_grad():
            o = getattr(m2, call)(x_enc=x, input_mask=mask)
        emb = getattr(o, "embeddings", None)
        rec = getattr(o, "reconstruction", None)
        sco = getattr(o, "anomaly_scores", None)
        print(f"  {call}: embeddings={None if emb is None else tuple(emb.shape)} "
              f"reconstruction={None if rec is None else tuple(rec.shape)} "
              f"anomaly_scores={None if sco is None else tuple(sco.shape)}", flush=True)
        have = emb if emb is not None else (rec if rec is not None else sco)
        rows.append({"series": "sp500_daily", "capability": call,
                     "embeddings_shape": None if emb is None else list(emb.shape),
                     "reconstruction_shape": None if rec is None else list(rec.shape),
                     "anomaly_scores_shape": None if sco is None else list(sco.shape),
                     "finite": bool(torch.isfinite(have).all()) if have is not None
                     else False})
        del m2
        torch.cuda.empty_cache()

    Path(a.out).write_text(json.dumps({"weights": str(w), "params": nparam,
                                       "head_audit": audit, "rows": rows}, indent=1))
    print(f"wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
