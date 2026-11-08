"""Environment / capability probe.

    python -m tsfm4finance.bench.doctor                  # imports only (cheap)
    python -m tsfm4finance.bench.doctor --weights        # also load + tiny forecast

It answers three questions that must not be answered by guesswork:
  1. can this python environment import the family at all (deps, python version);
  2. can the weights be fetched/loaded, and how much VRAM do they need;
  3. what does the model actually emit (point / quantile grid / samples), and at
     what cost per window -- the ETA table and the fair-comparison checks both
     depend on that.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import os
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reports" / "doctor.json"

FAMILIES = ["baseline", "timesfm25", "timesfm30", "chronos", "moirai", "moment",
            "ttm", "flowstate", "toto2", "tirex", "tirex2", "timemoe", "kronos"]

MODELS = {                            # probe targets: family -> hf repo id
    "timesfm25": "google/timesfm-2.5-200m-pytorch",
    "timesfm30": "google/timesfm-3.0-pytorch",
    "chronos": "amazon/chronos-2",
    "chronos_bolt": "amazon/chronos-bolt-small",
    "moirai": "Salesforce/moirai-2.0-R-small",
    "moment": "AutonLab/MOMENT-1-large",
    "ttm": "ibm-granite/granite-timeseries-ttm-r2",
    "flowstate": "ibm-granite/granite-timeseries-flowstate-r1",
    "toto2": "Datadog/Toto-2.0-22m",
    "tirex": "NX-AI/TiRex",
    "tirex2": "NX-AI/TiRex-2",
    "timemoe": "Maple728/TimeMoE-50M",
    "kronos": "NeoQuasar/Kronos-small",
}


def _series(name: str = "sp500_daily", n: int = 4, horizon: int = 21):
    d = REPO / "data" / "open" / name
    fr = pd.concat([pd.read_parquet(d / f"{f}.parquet")
                  for f in ("train", "validation", "test")])
    y = fr["close"].to_numpy(dtype=float)
    ctxs, truths = [], []
    for k in range(n):
        end = len(y) - (n - k) * horizon
        ctxs.append(y[max(0, end - 512):end])
        truths.append(y[end:end + horizon])
    return ctxs, truths


def _vram() -> float | None:
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.memory_allocated() / 1e6, 1)
    except Exception:                                     # noqa: BLE001
        pass
    return None


def _probe_import(fam: str) -> dict:
    out = {"family": fam}
    try:
        if fam == "baseline":
            import pmdarima, statsmodels                      # noqa: F401
        elif fam == "timesfm25":
            from timesfm import TimesFM_2p5_200M_torch        # noqa: F401
        elif fam == "timesfm30":
            from timesfm import TimesFM3Torch                 # noqa: F401
        elif fam == "chronos":
            import chronos                                    # noqa: F401
            out["pipelines"] = [x for x in dir(chronos) if x.endswith("Pipeline")]
        elif fam == "moirai":
            from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module  # noqa: F401
        elif fam == "moment":
            from momentfm import MOMENTPipeline               # noqa: F401
        elif fam in ("ttm",):
            from tsfm_public.models.tinytimemixer import (
                TinyTimeMixerForPrediction)                   # noqa: F401
        elif fam == "flowstate":
            import tsfm_public.models.flowstate as fs
            out["exports"] = [x for x in dir(fs) if not x.startswith("_")][:8]
        elif fam == "toto2":
            from toto2.model import Toto2Model, Toto2ModelConfig  # noqa: F401
        elif fam == "tirex":
            import tirex                                    # noqa: F401
        elif fam == "tirex2":
            import tirex2                                   # noqa: F401
        elif fam == "timemoe":
            from transformers import AutoModelForCausalLM, AutoConfig  # noqa: F401
        elif fam == "kronos":
            import sys
            sys.path.insert(0, str(REPO / "third_party" / "Kronos"))
            from model.kronos import Kronos, KronosTokenizer, KronosPredictor  # noqa: F401
        out["import"] = "ok"
    except Exception as e:                                # noqa: BLE001
        out["import"] = f"fail: {type(e).__name__}: {str(e)[:200]}"
    return out


def _probe_forecast(fam: str, device: str = "cuda") -> dict:
    """Load weights and run 4 disjoint windows at h=21; report shapes + timing."""
    ctxs, truths = _series()
    h = len(truths[0])
    t0 = time.time()
    out: dict = {"n_windows": len(ctxs), "horizon": h}
    if fam == "timesfm25":
        from timesfm import TimesFM_2p5_200M_torch, ForecastConfig
        f = TimesFM_2p5_200M_torch(ForecastConfig(per_core_batch_size=8))
        f.load_checkpoint("google/timesfm-2.5-200m-pytorch", device=device)
        pts, qs = f.forecast([(c, []) for c in ctxs], h)
        out.update(point_shape=np.asarray(pts).shape,
                   quantile_shape=(np.asarray(qs).shape if qs is not None else None))
    elif fam == "timesfm30":
        try:
            from timesfm import TimesFM3Forecaster
            fc = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch",
                                                    device=device, per_core_batch_size=4)
            o = fc.predict_batch(ctxs, horizon=h, return_quantiles=True)
            o = list(o)
            out.update(via="TimesFM3Forecaster",
                       point_shape=np.asarray(o[0].forecast).shape,
                       quantile_shape=(np.asarray(o[0].quantiles).shape
                                       if getattr(o[0], "quantiles", None) is not None else None))
        except Exception as e:                            # noqa: BLE001
            out["TimesFM3Forecaster"] = f"fail: {str(e)[:160]}"
            from timesfm import TimesFM3Torch
            m = TimesFM3Torch.from_pretrained("google/timesfm-3.0-pytorch")
            out.update(via="TimesFM3Torch(module only)", module_params=sum(
                p.numel() for p in m.parameters()))
    elif fam == "chronos":
        import chronos
        p = chronos.Chronos2Pipeline.from_pretrained(MODELS["chronos"], device_map=device,
                                                     dtype="bfloat16")
        q = p.predict(ctxs, prediction_length=h)
        out.update(out_shape=tuple(np.asarray(q).shape),
                   quantile_levels=getattr(getattr(p, "model", None), "quantiles", None))
    elif fam == "chronos_bolt":
        import chronos
        p = chronos.ChronosBoltPipeline.from_pretrained(MODELS["chronos_bolt"],
                                                        device_map=device, dtype="bfloat16")
        q = p.predict(ctxs, prediction_length=h)
        out.update(out_shape=tuple(np.asarray(q).shape),
                   quantile_levels=list(p.model.quantiles) if hasattr(p, "model") else None)
    elif fam == "moirai":
        import torch
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        mod = Moirai2Module.from_pretrained(MODELS["moirai"])
        mod = mod.to(device)
        f = Moirai2Forecast(module=mod, prediction_length=h, target_dim=1,
                            feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
                            context_length=512)
        pt = torch.tensor(np.stack(ctxs), dtype=torch.float32, device=device)[:, :, None]
        obs = torch.ones_like(pt, dtype=torch.bool)
        pad = torch.zeros(pt.shape[:2], dtype=torch.bool, device=device)
        with torch.no_grad():
            q = f(pt, obs, pad)
        cfg = getattr(mod, "config", None) or getattr(mod, "hparams", None)
        out.update(out_shape=tuple(q.shape), n_quantiles=int(q.shape[1]),
                   quantile_levels=getattr(cfg, "quantiles", None),
                   patch_size=int(getattr(mod.patch_size, "item", lambda: mod.patch_size)())
                   if not isinstance(getattr(mod, "patch_size", 1), int) else int(mod.patch_size))
    elif fam == "moment":
        from momentfm import MOMENTPipeline
        import torch
        m = MOMENTPipeline.from_pretrained(MODELS["moment"], model_kwargs={
            "task_name": "forecasting", "n_channels": 1, "n_pred_tasks": h})
        m.init()
        m = m.to(device)
        x = torch.tensor(np.stack(ctxs)[:, None, :], dtype=torch.float32, device=device)
        with torch.no_grad():
            o = m(x_enc=x)
        out.update(out_shape=tuple(o.prediction.shape),
                   pred_h=tuple(o.prediction.shape[-2:]))
    elif fam == "ttm":
        import torch
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
        m = TinyTimeMixerForPrediction.from_pretrained(MODELS["ttm"]).to(device)
        pv = torch.tensor(np.stack(ctxs)[:, :, None], dtype=torch.float32, device=device)
        with torch.no_grad():
            o = m(past_values=pv)
        out.update(mean_shape=tuple(o.prediction_outputs.shape),
                   quantile_shape=(tuple(o.quantiles.shape)
                                   if getattr(o, "quantiles", None) is not None else None),
                   quantile_levels=getattr(m.config, "quantiles", None))
    elif fam == "flowstate":
        import tsfm_public.models.flowstate as fs
        out["exports"] = [x for x in dir(fs) if not x.startswith("_")][:12]
        out["note"] = "pick the forecasting class from exports, then re-probe"
    elif fam == "toto2":
        import torch
        from toto2.model import Toto2Model
        m = Toto2Model.from_pretrained(MODELS["toto2"]).to(device)
        ctx = torch.tensor(np.stack(ctxs)[:, None, :], dtype=torch.float32, device=device)
        out["config_attrs"] = {k: v for k, v in m.config.__dict__.items()
                               if any(s in k.lower() for s in ("quant", "patch", "context", "d_model"))}
        with torch.no_grad():
            o = m(target=ctx, target_mask=torch.ones_like(ctx, dtype=torch.bool),
                  cpm_mask=torch.zeros_like(ctx, dtype=torch.bool),
                  series_ids=torch.arange(len(ctxs), device=device)[:, None].expand(-1, 1),
                  num_return_steps=h)
        out.update(loc_scale=tuple(np.asarray(o.loc.detach().cpu()).shape),
                   quantiles_shape=(tuple(np.asarray(o.quantiles.detach().cpu()).shape)
                                    if o.quantiles is not None else None))
    elif fam == "tirex":
        import tirex
        m = tirex.load_model(MODELS["tirex"], device=device)
        q = m.forecast(ctxs, output_type="numpy", batch_size=8)
        out.update(out_shape=tuple(np.asarray(q).shape),
                   max_context=getattr(m, "max_context_length", None))
    elif fam == "tirex2":
        import tirex2
        m = tirex2.load_model(ckpt_path=MODELS["tirex2"], device=device)
        q = m.forecast([{"target": c} for c in ctxs], prediction_length=h,
                       output_type="numpy", batch_size=8)
        out.update(out_shape=tuple(np.asarray(q).shape))
    elif fam == "timemoe":
        import torch
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(MODELS["timemoe"], trust_remote_code=True)
        m = m.to(device)
        m.eval()
        inp = [torch.tensor(c, dtype=torch.float32, device=device) for c in ctxs]
        with torch.no_grad():
            try:
                q = m.forecast(inputs=inp, forecast_horizon=h, count=4)
                out.update(via="forecast()", out_shape=tuple(np.asarray(q[0]).shape)
                           if isinstance(q, (list, tuple)) else tuple(np.asarray(q).shape))
            except Exception as e:                        # noqa: BLE001
                out["forecast()"] = f"fail: {str(e)[:160]}"
                out["methods"] = [x for x in dir(m) if x in ("forecast", "generate")]
    elif fam == "kronos":
        import sys
        sys.path.insert(0, str(REPO / "third_party" / "Kronos"))
        from model.kronos import Kronos, KronosTokenizer, KronosPredictor
        tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        m = Kronos.from_pretrained(MODELS["kronos"])
        pr = KronosPredictor(m, tok, device=device, max_context=512)
        df = pd.DataFrame({k: ctxs[0] for k in ("open", "high", "low", "close")})
        xt = pd.Series(pd.date_range("2020-01-01", periods=len(ctxs[0])))
        yt = pd.Series(pd.date_range("2020-01-01", periods=h))
        p = pr.predict(df=df, x_timestamp=xt, y_timestamp=yt, pred_len=h,
                       T=1.0, top_p=0.9, sample_count=1)
        out.update(via="KronosPredictor", out_shape=tuple(np.asarray(p).shape),
                   columns=list(p.columns))
    else:                                                 # baseline family
        from tsfm4finance.bench.engines.baselines import NaiveEngine
        pr = NaiveEngine().predict(ctxs, h)
        out.update(point_shape=pr.point.shape, quantile_shape=None)
    out["seconds"] = round(time.time() - t0, 2)
    out["vram_mb"] = _vram()
    out["truth_shape"] = np.asarray(truths).shape
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="probe model environments/weights")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--weights", action="store_true", help="also load weights and forecast")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--python", default=None, help="label recorded in the report")
    ap.add_argument("--weights-root", default=os.environ.get("TSFM_WEIGHTS_ROOT", ""),
                    help="directory holding git-cloned / hf-downloaded models")
    a = ap.parse_args()
    if a.weights_root:
        os.environ["TSFM_WEIGHTS_ROOT"] = a.weights_root
    import sys
    report = {"python": sys.version.split()[0], "device": a.device,
              "venv": a.python or str(Path(sys.executable).parent.parent.name),
              "checks": {}}
    for fam in [f.strip() for f in a.families.split(",")]:
        entry = _probe_import(fam)
        if a.weights and entry.get("import") == "ok":
            try:
                entry["forecast"] = _probe_forecast(fam, a.device)
            except Exception as e:                        # noqa: BLE001
                entry["forecast"] = {"fail": f"{type(e).__name__}: {str(e)[:300]}",
                                     "trace": traceback.format_exc()[-800:]}
                import torch
                torch.cuda.empty_cache() if __import__("torch").cuda.is_available() else None
        report["checks"][fam] = entry
        print(f"== {fam}: {json.dumps(entry, default=str)[:600]}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    prev.update(report)
    OUT.write_text(json.dumps(prev, indent=1, default=str))


if __name__ == "__main__":
    main()
