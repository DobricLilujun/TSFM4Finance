"""Engine smoke test: 4 disjoint windows, one horizon, one series.

Run it per environment before launching a sweep; it is also the regression test
that a library upgrade (new chronos/uni2ts/toto release) has not silently changed
an output layout under us.

    .venv2/bin/python    -m tsfm4finance.bench.smoke --models naive,chronos-2,timesfm-3.0
    .venv-moirai/bin/python -m tsfm4finance.bench.smoke --models moirai-2.0-R-small
    python -m tsfm4finance.bench.smoke --all-tiers A
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import os
import numpy as np

from .config import load_grid
from .engines import build_engine
from .protocol import build_windows

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reports" / "engine_smoke.json"


def windows_for(series: str, horizon: int, n: int, span: str = "test"):
    from tsfm4finance.bench.download.common import load_dataset
    tr, va, te, meta = load_dataset(series)
    wins, info = build_windows(tr, va, te, horizon=horizon, target=meta.get("target", "close"),
                               max_context=512, span=span)
    if not wins:                                  # short datasets: use the whole tail
        wins, info = build_windows(tr, va, te, horizon=horizon, target="close",
                                   max_context=512, span="val+test")
    return wins[-n:], info, meta


def score(win_list, pred, train_hist, info, horizon) -> dict:
    """Score through the same path the sweep uses, then add the raw shapes."""
    from .scoring import score_forecast
    from .engines.baselines import NaiveEngine
    naive = NaiveEngine().predict([w.context for w in win_list], horizon).point
    cell = score_forecast(win_list, pred, info=info, train_hist=train_hist, horizon=horizon,
                          baseline_point=naive, model_key="smoke")
    out = {k: v for k, v in cell["aggregate"].items() if not k.endswith("__std")}
    out["shape_point"] = list(pred.point.shape)
    out["shape_quantiles"] = list(pred.quantiles.shape) if pred.quantiles is not None else None
    fin = cell["finance"]
    for k in ("dm_t_vs_naive", "dm_p_vs_naive", "pit_mean", "pit_ks",
              "mean_abs_coverage_gap", "kupiec_p", "vol_zscore_ratio", "ndp"):
        if k in fin:
            out[k] = fin[k]
    if "ndp" in cell:
        out["ndp"] = cell["ndp"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="")
    ap.add_argument("--all-tiers", default="", help="e.g. A or A,B")
    ap.add_argument("--series", default="sp500_daily")
    ap.add_argument("--horizon", type=int, default=21)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", default="fp32")
    ap.add_argument("--weights-root", default=os.environ.get("TSFM_WEIGHTS_ROOT", ""),
                    help="directory holding git-cloned / hf-downloaded models")
    a = ap.parse_args()
    if a.weights_root:
        os.environ["TSFM_WEIGHTS_ROOT"] = a.weights_root

    grid = load_grid()
    tiers = tuple(t.strip() for t in a.all_tiers.split(",") if t.strip()) or None
    only = tuple(m.strip() for m in a.models.split(",") if m.strip()) or None
    specs = grid.models(tiers=tiers, only=only)
    if not only:      # with --all-tiers, keep only the models this interpreter can import
        me = Path(__file__).resolve()
        specs = {k: v for k, v in specs.items()
                 if Path(grid.python_for(v)).resolve().parent.parent == me.parents[2]}
    report = {}
    wins = info = meta = None
    for key, spec in specs.items():
        env_py = grid.python_for(spec)
        if Path(env_py).resolve() != Path(__file__).resolve().parents[3] / "x":
            pass
        try:
            if wins is None:
                wins, info, meta = windows_for(a.series, a.horizon, a.n)
                from tsfm4finance.bench.download.common import load_dataset
                _tr, _va, _te, _m = load_dataset(a.series)
                train_hist = _tr[_m.get("target", "close")].to_numpy(dtype=float)
            t0 = time.time()
            eng = build_engine(spec, device=a.device, amp=a.amp,
                               ctx_cap=min(512, spec.max_context))
            panels = [w.panel for w in wins] if "panel_in" in spec.caps else None
            pred = eng.predict([w.context for w in wins], a.horizon, panels=panels)
            rep = {"ok": True, "seconds": round(time.time() - t0, 2),
                   "n_windows": len(wins), "horizon": a.horizon,
                   "metrics": score(wins, pred, train_hist, info, a.horizon),
                   "meta": pred.meta,
                   "no_leakage": info.get("no_leakage"), "env": Path(grid.python_for(spec)).parent.parent.name}
            if "embed" in spec.caps:
                try:
                    rep["embed_shape"] = list(np.asarray(eng.encode([w.context for w in wins])).shape)
                except Exception as e:                            # noqa: BLE001
                    rep["embed_shape"] = f"fail: {str(e)[:120]}"
            eng.close()
        except Exception as e:                                    # noqa: BLE001
            rep = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}",
                   "trace": traceback.format_exc()[-1200:]}
        report[key] = rep
        print(f"{key:<22} {json.dumps(rep, default=str)[:300]}", flush=True)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    prev.update(report)
    OUT.write_text(json.dumps(prev, indent=1, default=str))


if __name__ == "__main__":
    main()
