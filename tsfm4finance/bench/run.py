"""Run the TimesFM benchmark over the arena's real financial datasets (spec §4.1).

Stages (each writes into reports/timesfm_benchmark_results.json as it goes, so an
interrupted run keeps its work):

  finetune   full fine-tune TimesFM 2.5-200M per dataset at K = 16 / 64 / full train split
  forecast   rolling-origin backtest, h in {1,5,12,21,63}, naive / snaive / MA / ARIMA /
             Chronos-2 / TimesFM(zero-shot @ ctx 128/256/512) / TimesFM fine-tuned (16/64/full)
  classify   *_dir5 and *_volreg: forecast-feature head, embedding probe, statistical baselines
  anomaly    *_extreme: statistical MAD z-score, TimesFM predictive-band violation, probe
  trading    *_trade: signal -> position -> cost-aware backtest vs buy&hold / momentum
  leaderboard append spec-§4.1 rows (h=21) to backend/leaderboard.json
  report     render reports/02_timesfm_benchmark.md from the JSON

Usage:  python -m tsfm4finance.bench.run [--stage NAME ...] [--limit N]
"""
from __future__ import annotations

import argparse
import os
import shutil
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from tsfm4finance.paths import DATA
from tsfm4finance.core.schemas import DatasetMeta
from tsfm4finance.core import metrics as M
from tsfm4finance.bench import tasks as T
from tsfm4finance.bench import protocol as P
from tsfm4finance.bench import finetune as FT
from tsfm4finance.bench import probe as PR
from tsfm4finance.bench import trading as TR

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports"
RESULTS_PATH = REPORT_DIR / "timesfm_benchmark_results.json"
CKPT_DIR = REPORT_DIR / "ft_ckpts"
LEADERBOARD = ROOT / "tsfm4finance" / "backend" / "leaderboard.json"
LOCK = REPORT_DIR / ".bench.lock"
FT_BS = 8

HORIZONS = P.HORIZONS                       # [1, 5, 12, 21, 63]
PRIMARY_H = 21
FT_VARIANTS = [("fs16", 16), ("fs64", 64), ("full", None)]
FT_LR = 1e-5                                # chosen by the LR pilot (see report §fine-tuning)
CONTEXTS = [128, 256, 512]                  # zero-shot context ablation
FT_CONTEXT = FT.CONTEXT                     # 256, matched against zero-shot @256


# --------------------------------------------------------------------------- #
# persistence helpers
# --------------------------------------------------------------------------- #
def load_results() -> dict:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return {"meta": {"protocol": "spec-4.1", "horizons": HORIZONS,
                     "primary_horizon": PRIMARY_H, "contexts": CONTEXTS,
                     "ft_lr": FT_LR, "ft_context": FT_CONTEXT},
            "finetune": {}, "forecast": {}, "classify": {}, "anomaly": {}, "trading": {}}


def save_results(res: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(res, indent=1, default=_native))


def _native(o):
    """JSON-safe plain types; non-finite floats become null so the leaderboard file
    stays strictly valid JSON (Python reprs with bare `nan` do not)."""
    if isinstance(o, dict):
        return {str(k): _native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_native(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_native(v) for v in o.tolist()]
    if isinstance(o, bool):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if np.isfinite(f) else None
    if o is None or isinstance(o, str):
        return o
    return str(o)


def load_ds(name: str) -> tuple[DatasetMeta, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = DATA / "open" / name
    meta = DatasetMeta.model_validate_json((base / "meta.json").read_text())
    return (meta,
            pd.read_parquet(base / "train.parquet"),
            pd.read_parquet(base / "validation.parquet"),
            pd.read_parquet(base / "test.parquet"))


def windows_with_labels(name: str, horizon: int, max_context: int = P.MAX_CONTEXT,
                        train_stride: int | None = None):
    """Eval windows (held-out span) + fit windows (train span, target strictly in train)."""
    meta, tr, va, te = load_ds(name)
    full = pd.concat([tr, va, te], ignore_index=True)
    close = full["close"].to_numpy(dtype=float)
    labels = full["label"].to_numpy(dtype=float) if "label" in full.columns else None
    n_train = len(tr)

    ev, info = P.build_windows(tr, va, te, horizon=horizon, target="close",
                               max_context=max_context)
    # supervised tasks: the last `horizon` rows have no computable label (the forward
    # window runs past the end of the data) -> those windows are dropped, not coerced
    if labels is not None:
        ev = [w for w in ev if np.isfinite(labels[w.origin])]
    ev_idx = [w.origin for w in ev]
    st = train_stride or horizon
    fit_idx = [t for t in range(P.MIN_CONTEXT, n_train - horizon - 1, st)
               if labels is None or np.isfinite(labels[t])]
    fit_ctx = [close[max(0, t - max_context):t] for t in fit_idx]
    ev_ctx = [w.context for w in ev]
    return {"meta": meta, "close": close, "labels": labels, "n_train": n_train,
            "eval_ctx": ev_ctx, "eval_idx": ev_idx, "eval_truth": [w.truth for w in ev],
            "fit_ctx": fit_ctx, "fit_idx": fit_idx, "info": info}


# --------------------------------------------------------------------------- #
# stage: fine-tune
# --------------------------------------------------------------------------- #
def stage_finetune(res: dict, limit: int | None = None) -> None:
    res["finetune"] = res.get("finetune", {})
    series = T.REAL_SERIES[:limit] if limit else T.REAL_SERIES
    for base, _dom in series:
        _, tr, _va, _te = load_ds(base)
        close = tr["close"].to_numpy(dtype=float)
        for tag, K in FT_VARIANTS:
            key = f"{base}:{tag}"
            out = CKPT_DIR / f"{base}_{tag}"
            # checkpoints are transient artifacts: retrain only while some consumer
            # (the forecast grid or the trading calendar) still needs this variant
            needs = (any(f"{base}:h{h}:timesfm_ft:{tag}:{base}" not in res.get("forecast", {})
                         for h in HORIZONS)
                     or f"model_{tag}" not in res.get("trading", {}).get(f"{base}_trade", {}))
            if key in res["finetune"] and not needs:
                print(f"[finetune] skip {key} (done; checkpoint reclaimed)", flush=True)
                continue
            print(f"[finetune] {key} K={K} lr={FT_LR} ...", flush=True)
            t0 = time.time()
            try:
                r = FT.finetune(close, out, dataset=base, k_windows=K, lr=FT_LR,
                                context=FT_CONTEXT, batch_size=FT_BS, epochs=args.ft_epochs,
                                device=args_ft["device"], amp=args_ft["amp"])
                res["finetune"][key] = r.to_dict()
            except Exception as e:
                res["finetune"][key] = {"error": f"{type(e).__name__}: {e}"}
                traceback.print_exc()
            save_results(res)
            print(f"[finetune]   done in {time.time()-t0:.0f}s", flush=True)
            from tsfm4finance.core.models import TimesFMModel
            TimesFMModel.clear_cache()


# --------------------------------------------------------------------------- #
# stage: forecast
# --------------------------------------------------------------------------- #
_ENGINE_CACHE: dict[str, P.Engine] = {}


def _engines(names: list[str]) -> dict[str, P.Engine]:
    """Build engine instances, reusing the expensive ones (Chronos) across calls."""
    eng: dict[str, P.Engine] = {}
    for n in names:
        if n in _ENGINE_CACHE:
            eng[n] = _ENGINE_CACHE[n]
            continue
        if n == "naive":
            eng[n] = P.NaiveEngine()
        elif n == "seasonal_naive":
            eng[n] = P.SeasonalNaiveEngine(season=5)
        elif n == "moving_average":
            eng[n] = P.MovingAverageEngine(window=20)
        elif n == "arima":
            eng[n] = P.ARIMAEngine(order=(1, 1, 0))
        elif n == "chronos":
            eng[n] = P.ChronosEngine()
        elif n.startswith("timesfm@"):
            eng[n] = P.TimesFMEngine(max_context=int(n.split("@")[1]), name=n)
        elif n.startswith("timesfm_ft"):
            tag, base = n.split(":")[1], n.split(":")[2]
            eng[n] = P.TimesFMEngine(checkpoint=str(CKPT_DIR / f"{base}_{tag}"),
                                     max_context=FT_CONTEXT, name=n)
        else:
            raise ValueError(f"unknown engine {n!r}")
        if n != "naive":
            _ENGINE_CACHE[n] = eng[n]
    return eng


def forecast_model_list() -> list[str]:
    ms = ["naive", "seasonal_naive", "moving_average", "arima", "chronos"]
    ms += [f"timesfm@{c}" for c in CONTEXTS]
    ms += [f"timesfm_ft:{t}:{base}" for t, _ in FT_VARIANTS for base, _d in T.REAL_SERIES]
    return ms


def stage_forecast(res: dict, limit: int | None = None) -> None:
    res["forecast"] = res.get("forecast", {})
    series = T.REAL_SERIES[:limit] if limit else T.REAL_SERIES
    generic = ["naive", "seasonal_naive", "moving_average", "arima", "chronos"] + \
              [f"timesfm@{c}" for c in CONTEXTS]
    for base, _dom in series:
        meta, tr, va, te = load_ds(base)
        names = generic + [f"timesfm_ft:{t}:{base}" for t, _ in FT_VARIANTS]
        eng = _engines(names)
        for h in HORIZONS:
            W, info = P.build_windows(tr, va, te, horizon=h, target="close")
            if not W:
                continue
            for name in names:
                key = f"{base}:h{h}:{name}"
                if key in res["forecast"]:
                    continue
                t0 = time.time()
                try:
                    pts, qs = eng[name].predict_batch([w.context for w in W], h)
                    per = []
                    for i, w in enumerate(W):
                        out = M.PredictOutput(point=[float(x) for x in pts[i]])
                        if qs is not None:
                            out.quantiles = [[float(x) for x in row] for row in qs[i].T]
                        per.append(M.evaluate(out, pd.DataFrame({"close": w.truth}), meta,
                                              train_df=tr))
                    agg = P.aggregate(per)
                    agg["_secs"] = round(time.time() - t0, 1)
                    agg.update({f"proto_{k}": v for k, v in info.items() if k == "coverage"})
                    res["forecast"][key] = {"aggregate": agg, "per_window": per,
                                            "info": info}
                except Exception as e:
                    res["forecast"][key] = {"error": f"{type(e).__name__}: {e}"}
                    traceback.print_exc()
                save_results(res)
                print(f"[forecast] {key:52s} "
                      f"mase={res['forecast'][key].get('aggregate',{}).get('mase', float('nan')):.3f} "
                      f"({res['forecast'][key].get('_secs', 0)}s)", flush=True)
                _gc_checkpoints(res)
            from tsfm4finance.core.models import TimesFMModel
            TimesFMModel.clear_cache()


# --------------------------------------------------------------------------- #
# stage: valcheck -- predict on the VALIDATION split (never seen by training)
# --------------------------------------------------------------------------- #
VAL_MODELS = ["naive", "seasonal_naive", "arima", "chronos", "timesfm@256", "timesfm@512"]
VAL_HORIZONS = (5, 21)


def stage_valcheck(res: dict, limit: int | None = None, with_ft: bool = True) -> None:
    """Same rolling protocol, evaluated strictly on the validation span.

    Answers the "did you actually hold data out?" question directly: fine-tuning windows
    are sampled from the train span only, and this stage predicts on the *validation*
    span -- a period that lies between training and test and that no model was tuned on.
    """
    res["valcheck"] = res.get("valcheck", {})
    series = T.REAL_SERIES[:limit] if limit else T.REAL_SERIES
    from tsfm4finance.core.models import TimesFMModel
    for base, _dom in series:
        meta, tr, va, te = load_ds(base)
        done = all(f"{base}:h{h}:{m}" in res["valcheck"]
                   for h in VAL_HORIZONS for m in VAL_MODELS)
        if done and not with_ft:
            print(f"[valcheck] skip {base} (done)", flush=True)
            continue
        t0 = time.time()
        # --- fine-tuned model first: full-parameter training needs the whole GPU, so
        # release the inference engines (Chronos ~1 GB, TimesFM ~1 GB) before it starts
        if with_ft:
            ck = CKPT_DIR / f"{base}_full"
            ft_rows_done = all(f"{base}:h{h}:timesfm_ft:full:{base}" in res["valcheck"]
                               and "aggregate" in res["valcheck"][f"{base}:h{h}:timesfm_ft:full:{base}"]
                               for h in VAL_HORIZONS)
            if not ft_rows_done and not (ck / "model.safetensors").exists():
                shutil.rmtree(ck, ignore_errors=True)
                import torch
                torch.cuda.empty_cache()
                _ENGINE_CACHE.pop("chronos", None)
                print(f"[valcheck] retraining {base}:full for the val check ...", flush=True)
                try:
                    FT.finetune(tr["close"].to_numpy(dtype=float), ck, dataset=base,
                                k_windows=None, lr=FT_LR, context=FT_CONTEXT,
                                batch_size=FT_BS, epochs=args.ft_epochs,
                                device=args_ft["device"], amp=args_ft["amp"])
                except Exception as e:
                    print(f"[valcheck] fine-tune failed for {base}: {e}", flush=True)
                    shutil.rmtree(ck, ignore_errors=True)
                    for h in VAL_HORIZONS:
                        res["valcheck"][f"{base}:h{h}:timesfm_ft:full:{base}"] = {
                            "unavailable": f"fine-tune failed: {type(e).__name__}"}
                    save_results(res)
        # zero-shot / statistical engines on the validation span
        for h in VAL_HORIZONS:
            W, info = P.build_windows(tr, va, te, horizon=h, target="close", span="val")
            tr_df = tr
            for name in VAL_MODELS:
                key = f"{base}:h{h}:{name}"
                if key in res["valcheck"]:
                    continue
                try:
                    eng = _engines([name])[name]
                    pts, qs = eng.predict_batch([w.context for w in W], h)
                    per = []
                    for i, w in enumerate(W):
                        out = M.PredictOutput(point=[float(x) for x in pts[i]])
                        if qs is not None:
                            out.quantiles = [[float(x) for x in row] for row in qs[i].T]
                        per.append(M.evaluate(out, pd.DataFrame({"close": w.truth}), meta,
                                              train_df=tr_df))
                    agg = P.aggregate(per)
                    agg["_secs"] = round(time.time() - t0, 1)
                    res["valcheck"][key] = {"aggregate": agg, "info": info}
                except Exception as e:
                    res["valcheck"][key] = {"error": f"{type(e).__name__}: {e}"}
                    traceback.print_exc()
                save_results(res)
        # evaluate the fine-tuned model on the same validation windows
        if with_ft:
            ck = CKPT_DIR / f"{base}_full"
            if (ck / "model.safetensors").exists():
                TimesFMModel.clear_cache()
                _ENGINE_CACHE.pop("chronos", None)
                for h in VAL_HORIZONS:
                    W, info = P.build_windows(tr, va, te, horizon=h, target="close", span="val")
                    key = f"{base}:h{h}:timesfm_ft:full:{base}"
                    if key not in res["valcheck"]:
                        try:
                            eng = P.TimesFMEngine(checkpoint=str(ck), max_context=FT_CONTEXT, name=key)
                            pts, qs = eng.predict_batch([w.context for w in W], h)
                            per = []
                            for i, w in enumerate(W):
                                out = M.PredictOutput(point=[float(x) for x in pts[i]])
                                if qs is not None:
                                    out.quantiles = [[float(x) for x in row] for row in qs[i].T]
                                per.append(M.evaluate(out, pd.DataFrame({"close": w.truth}), meta,
                                                      train_df=tr))
                            agg = P.aggregate(per)
                            agg["_secs"] = round(time.time() - t0, 1)
                            res["valcheck"][key] = {"aggregate": agg, "info": info}
                        except Exception as e:
                            res["valcheck"][key] = {"error": f"{type(e).__name__}: {e}"}
                            traceback.print_exc()
                    save_results(res)
                shutil.rmtree(ck, ignore_errors=True)      # transient weights only
        TimesFMModel.clear_cache()
        vals = [res["valcheck"][f"{base}:h21:{m}"]["aggregate"].get("mase", float("nan"))
                for m in VAL_MODELS + [f"timesfm_ft:full:{base}"]
                if f"{base}:h21:{m}" in res["valcheck"]
                and "aggregate" in res["valcheck"][f"{base}:h21:{m}"]]
        if vals:
            print(f"[valcheck] {base:24s} h21 MASE {min(vals):.3f}-{max(vals):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        else:
            print(f"[valcheck] {base:24s} no successful val rows", flush=True)


def _gc_checkpoints(res: dict) -> None:
    """Reclaim fine-tune checkpoints whose forecast rows are all recorded.

    A 200M-parameter checkpoint is ~700 MB; keeping 36 of them fills the disk, and
    the benchmark outputs (not the weights) are what the report and paper consume.
    """
    if not CKPT_DIR.exists():
        return
    for d in sorted(CKPT_DIR.iterdir()):
        if not d.is_dir():
            continue
        base, tag = d.name.rsplit("_", 1)
        rows = [f"{base}:h{h}:timesfm_ft:{tag}:{base}" for h in HORIZONS]
        if all(k in res.get("forecast", {}) and "aggregate" in res["forecast"][k] for k in rows):
            shutil.rmtree(d, ignore_errors=True)
            print(f"[forecast] reclaimed checkpoint {d.name}", flush=True)


# --------------------------------------------------------------------------- #
# stage: classify
# --------------------------------------------------------------------------- #
def stage_classify(res: dict, limit: int | None = None) -> None:
    res["classify"] = res.get("classify", {})
    series = T.REAL_SERIES[:limit] if limit else T.REAL_SERIES
    from tsfm4finance.core.models import TimesFMModel
    for base, _dom in series:
        for suffix in ("dir5", "volreg"):
            name = f"{base}_{suffix}"
            if name in res["classify"]:
                continue
            t0 = time.time()
            entry: dict = {}
            try:
                W = windows_with_labels(name, horizon=5 if suffix == "dir5" else 21)
                meta = W["meta"]
                h = meta.horizon
                labels = W["labels"]
                y_fit = np.array([labels[t] for t in W["fit_idx"]], dtype=int)
                y_ev = np.array([labels[t] for t in W["eval_idx"]], dtype=int)
                entry["n_fit"] = len(y_fit)
                entry["n_eval"] = len(y_ev)
                entry["eval_label_counts"] = {int(k): int(v) for k, v in
                                              zip(*np.unique(y_ev, return_counts=True))}

                # ---- statistical features -> logistic / gbm ----
                X_fit = np.array([PR.stat_features(W["close"][:t + 1]) for t in W["fit_idx"]])
                X_ev = np.array([PR.stat_features(W["close"][:t + 1]) for t in W["eval_idx"]])
                for kind in ("logistic", "gbm"):
                    clf = PR.safe_fit(kind, X_fit, y_fit)
                    if clf is None:
                        entry[f"stat_{kind}"] = {"unavailable": "fit split has a single class"}
                        continue
                    yp = clf.predict(X_ev)
                    ys = None
                    if hasattr(clf, "predict_proba"):
                        pr = clf.predict_proba(X_ev)
                        ys = pr[:, 1] if meta.n_classes == 2 else pr
                    entry[f"stat_{kind}"] = PR.eval_classifier(y_ev, yp, ys, meta.n_classes)
                # ---- majority reference ----
                maj = int(np.bincount(y_fit).argmax())
                entry["majority"] = PR.eval_classifier(y_ev, np.full_like(y_ev, maj), None,
                                                       meta.n_classes)

                # ---- TimesFM routes (forecast features + embedding probe) ----
                eng = P.TimesFMEngine(max_context=FT_CONTEXT)
                pts_f, qs_f = eng.predict_batch(W["fit_ctx"], h)
                pts_e, qs_e = eng.predict_batch(W["eval_ctx"], h)
                Ff = np.array([PR.forecast_features(c, pts_f[i], None if qs_f is None else qs_f[i], h)
                               for i, c in enumerate(W["fit_ctx"])])
                Fe = np.array([PR.forecast_features(c, pts_e[i], None if qs_e is None else qs_e[i], h)
                               for i, c in enumerate(W["eval_ctx"])])
                for kind in ("logistic", "gbm"):
                    clf = PR.safe_fit(kind, Ff, y_fit)
                    if clf is None:
                        entry[f"timesfm_forecastfeat_{kind}"] = {"unavailable": "fit split has a single class"}
                        continue
                    yp = clf.predict(Fe)
                    ys = clf.predict_proba(Fe) if hasattr(clf, "predict_proba") else None
                    if ys is not None and meta.n_classes == 2:
                        ys = ys[:, 1]
                    entry[f"timesfm_forecastfeat_{kind}"] = PR.eval_classifier(y_ev, yp, ys,
                                                                               meta.n_classes)
                # embedding probe
                embf = np.array([eng.encode(c, max_patches=8) for c in W["fit_ctx"]])
                embe = np.array([eng.encode(c, max_patches=8) for c in W["eval_ctx"]])
                for kind in ("logistic", "gbm"):
                    clf = PR.safe_fit(kind, embf, y_fit)
                    if clf is None:
                        entry[f"timesfm_probe_{kind}"] = {"unavailable": "fit split has a single class"}
                        continue
                    yp = clf.predict(embe)
                    ys = clf.predict_proba(embe) if hasattr(clf, "predict_proba") else None
                    if ys is not None and meta.n_classes == 2:
                        ys = ys[:, 1]
                    entry[f"timesfm_probe_{kind}"] = PR.eval_classifier(y_ev, yp, ys,
                                                                        meta.n_classes)
                entry["_secs"] = round(time.time() - t0, 1)
            except Exception as e:
                entry = {"error": f"{type(e).__name__}: {e}"}
                traceback.print_exc()
            res["classify"][name] = entry
            save_results(res)
            TimesFMModel.clear_cache()
            print(f"[classify] {name:32s} {_secs_str(entry)}", flush=True)


def _secs_str(entry: dict) -> str:
    if "error" in entry:
        return "ERROR " + entry["error"][:60]
    best = max(((k, v.get("f1", 0)) for k, v in entry.items() if isinstance(v, dict)
                and "f1" in v), key=lambda kv: kv[1], default=("?", 0))
    return f"best {best[0]} f1={best[1]:.3f} ({entry.get('_secs', 0)}s)"


# --------------------------------------------------------------------------- #
# stage: anomaly
# --------------------------------------------------------------------------- #
def stage_anomaly(res: dict, limit: int | None = None) -> None:
    res["anomaly"] = res.get("anomaly", {})
    series = T.REAL_SERIES[:limit] if limit else T.REAL_SERIES
    from tsfm4finance.core.models import TimesFMModel
    for base, _dom in series:
        name = f"{base}_extreme"
        if name in res["anomaly"]:
            continue
        entry: dict = {}
        t0 = time.time()
        try:
            W = windows_with_labels(name, horizon=1)
            close, labels = W["close"], W["labels"]
            y_fit = np.array([labels[t] for t in W["fit_idx"]], dtype=int)
            y_ev = np.array([labels[t] for t in W["eval_idx"]], dtype=int)
            entry["base_rate_eval"] = float(np.mean(y_ev))
            entry["n_fit"], entry["n_eval"] = len(y_fit), len(y_ev)

            entry["n_pos_eval"] = int(np.sum(y_ev))
            entry["n_pos_fit"] = int(np.sum(y_fit))

            # (0) ORACLE reference: the label-generating rule itself (median/MAD).
            # Kept only as a reference point -- it is not a competitor, it IS the label.
            def zscores(idx):
                out = []
                for t in idx:
                    hist = np.diff(np.log(np.maximum(close[:t], 1e-12)))
                    r = float(np.log(max(close[t], 1e-12) / max(close[t - 1], 1e-12)))
                    med = np.median(hist); mad = np.median(np.abs(hist - med))
                    out.append(abs(r - med) / (1.4826 * mad) if mad > 0 else 0.0)
                return np.array(out)
            zf, ze = zscores(W["fit_idx"]), zscores(W["eval_idx"])
            thr = _best_threshold(zf, y_fit)
            entry["oracle_label_rule"] = _anum(y_ev, (ze >= thr).astype(int), ze)

            # (1) fair statistical competitor: rolling mean/sigma z-score (different rule
            # from the median/MAD label), threshold chosen on the TRAIN split only.
            def zscores2(idx):
                out = []
                for t in idx:
                    hist = np.diff(np.log(np.maximum(close[:t], 1e-12)))
                    r = float(np.log(max(close[t], 1e-12) / max(close[t - 1], 1e-12)))
                    mu = np.mean(hist[-252:]); sd = np.std(hist[-252:], ddof=1)
                    out.append(abs(r - mu) / sd if sd > 0 else 0.0)
                return np.array(out)
            zf2, ze2 = zscores2(W["fit_idx"]), zscores2(W["eval_idx"])
            thr3 = _best_threshold(zf2, y_fit)
            entry["stat_zscore"] = _anum(y_ev, (ze2 >= thr3).astype(int), ze2)

            # TimesFM predictive-band violation (one-step forecast of y_t from t-1)
            eng = P.TimesFMEngine(max_context=FT_CONTEXT)

            def band_scores(ctx_list, idxs):
                pts, qs = eng.predict_batch(ctx_list, 1)
                out = []
                for i, t in enumerate(idxs):
                    y = close[t]
                    Q = qs[i, -1] if qs is not None else None
                    if Q is None:
                        out.append(0.0); continue
                    lo, hi = float(Q[0]), float(Q[-1])
                    w = max(hi - lo, 1e-9)
                    out.append(max((lo - y) / w, (y - hi) / w, 0.0))
                return np.array(out)
            # contexts for anomaly detection end at t-1 (predict y_t)
            ctx_fit = [close[max(0, t - 1 - FT_CONTEXT):t] for t in W["fit_idx"]]
            ctx_ev = [close[max(0, t - 1 - FT_CONTEXT):t] for t in W["eval_idx"]]
            bf, be = band_scores(ctx_fit, W["fit_idx"]), band_scores(ctx_ev, W["eval_idx"])
            thr2 = _best_threshold(bf, y_fit)
            entry["timesfm_bandviolation"] = _anum(y_ev, (be >= thr2).astype(int), be)

            # embedding probe on the same windows
            embf = np.array([eng.encode(c, max_patches=8) for c in ctx_fit])
            embe = np.array([eng.encode(c, max_patches=8) for c in ctx_ev])
            clf = PR.safe_fit("logistic", embf, y_fit)
            if clf is None:
                entry["timesfm_probe"] = {"unavailable": "train split has no extreme events"}
            else:
                p_e = clf.predict_proba(embe)[:, 1]
                entry["timesfm_probe"] = _anum(y_ev, (p_e >= 0.5).astype(int), p_e)
            entry["_secs"] = round(time.time() - t0, 1)
        except Exception as e:
            entry = {"error": f"{type(e).__name__}: {e}"}
            traceback.print_exc()
        res["anomaly"][name] = entry
        save_results(res)
        TimesFMModel.clear_cache()
        print(f"[anomaly ] {name:32s} {_secs_str(entry)}", flush=True)


def _best_threshold(score: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    cand = np.unique(score)
    if len(cand) > 60:
        cand = np.quantile(score, np.linspace(0.5, 0.99, 60))
    best, bs = 0.5, -1.0
    for c in cand:
        f = f1_score(y, (score >= c).astype(int), zero_division=0)
        if f > bs:
            best, bs = float(c), float(f)
    return best


def _anum(y: np.ndarray, yhat: np.ndarray, score: np.ndarray) -> dict:
    from tsfm4finance.core.metrics import f1_score, precision_recall, auc_roc, accuracy
    out = {"f1": f1_score(y, yhat, average="binary"),
           "accuracy": accuracy(y, yhat)}
    p, r = precision_recall(y, yhat)
    out["precision"], out["recall"] = p, r
    out["auc"] = auc_roc(y, score)
    out["flag_rate"] = float(np.mean(yhat))
    return out


# --------------------------------------------------------------------------- #
# stage: trading
# --------------------------------------------------------------------------- #
def stage_trading(res: dict, limit: int | None = None, h: int = 5) -> None:
    res["trading"] = res.get("trading", {})
    series = T.REAL_SERIES[:limit] if limit else T.REAL_SERIES
    from tsfm4finance.core.models import TimesFMModel
    for base, _dom in series:
        name = f"{base}_trade"
        if name in res["trading"]:
            continue
        entry: dict = {}
        t0 = time.time()
        try:
            W = windows_with_labels(name, horizon=h)
            close = W["close"]
            idx = W["eval_idx"]
            # realized h-day return of the window itself: buy at the close of the origin
            # day, sell at the last close inside the truth block. Non-overlapping origins
            # (stride = h) therefore partition the span with no double counting and no use
            # of prices beyond the truth block.
            fwd = np.array([float(np.log(max(float(tr[-1]), 1e-12) / max(close[t], 1e-12)))
                            for t, tr in zip(idx, W["eval_truth"])])
            eng = P.TimesFMEngine(max_context=FT_CONTEXT)
            pts, qs = eng.predict_batch(W["eval_ctx"], h)
            entry["n_trades"] = len(idx)
            entry["horizon_days"] = h
            entry["asset_buy_hold"] = {f"{int(cb)}bps": M.trading_metrics(
                np.ones(len(idx)), fwd, cost_bps=cb) for cb in M.TRADING_COST_GRID}
            entry["model"] = TR.trade_eval(W["eval_ctx"], pts,
                                           None if qs is None else [qs[i] for i in range(len(idx))],
                                           idx, close, fwd, h)
            entry["benchmarks"] = TR.benchmark_trades(close, idx, fwd, h)
            # fine-tuned signals on the same trade calendar; the ~700 MB checkpoints are
            # reclaimed per series so the run never accumulates 36 of them on disk
            for tag, _K in FT_VARIANTS:
                ck = CKPT_DIR / f"{base}_{tag}"
                if not ck.exists():
                    continue
                eng_ft = P.TimesFMEngine(checkpoint=str(ck), max_context=FT_CONTEXT,
                                         name=f"timesfm_ft:{tag}:{base}")
                p2, q2 = eng_ft.predict_batch(W["eval_ctx"], h)
                entry[f"model_{tag}"] = TR.trade_eval(
                    W["eval_ctx"], p2, None if q2 is None else [q2[i] for i in range(len(idx))],
                    idx, close, fwd, h)
                shutil.rmtree(ck, ignore_errors=True)
            entry["_secs"] = round(time.time() - t0, 1)
        except Exception as e:
            entry = {"error": f"{type(e).__name__}: {e}"}
            traceback.print_exc()
        res["trading"][name] = entry
        save_results(res)
        TimesFMModel.clear_cache()
        sh = entry.get("model", {}).get("sign", {}).get("10bps", {}).get("sharpe_net", float("nan"))
        print(f"[trading ] {name:32s} sign net Sharpe(10bps)={sh:.2f} ({entry.get('_secs',0)}s)",
              flush=True)


# --------------------------------------------------------------------------- #
# derived: normalized error vs the persistence forecast on identical windows
# --------------------------------------------------------------------------- #
def stage_derive(res: dict) -> None:
    """NDp = mean MAE(model) / mean MAE(naive/persistence) over the same windows.

    MASE's denominator is the in-sample train naive scale (standard, but small on
    low-volatility series), so NDp is the robust cross-asset normalizer: NDp < 1 means
    the model beats "keep the last price" on exactly the forecast task being scored.
    """
    out: dict = {}
    for key, val in res.get("forecast", {}).items():
        base, hstr, model = key.split(":", 2)
        if "aggregate" not in val:
            continue
        nk = f"{base}:{hstr}:naive"
        if nk not in res["forecast"] or "aggregate" not in res["forecast"][nk]:
            continue
        m_mae = val["aggregate"].get("mae", float("nan"))
        n_mae = res["forecast"][nk]["aggregate"].get("mae", float("nan"))
        out.setdefault(f"{base}:{hstr}", {})[model] = (
            float(m_mae / n_mae) if n_mae and np.isfinite(m_mae) and np.isfinite(n_mae) else float("nan"))
    res["derived"] = out
    save_results(res)
    print(f"[derive] normalized errors for {len(out)} (series,horizon) cells", flush=True)


# --------------------------------------------------------------------------- #
# stage: leaderboard
# --------------------------------------------------------------------------- #
def stage_leaderboard(res: dict, horizon: int = PRIMARY_H) -> int:
    if not LEADERBOARD.exists():
        print("[leaderboard] missing file", flush=True)
        return 0
    recs = json.loads(LEADERBOARD.read_text())
    for r in recs:                          # tag the old single-window rows
        r.setdefault("protocol", "legacy-single-window")
    existing = {(r.get("dataset"), r.get("model"), r.get("protocol")) for r in recs}
    added = 0
    for key, val in res.get("forecast", {}).items():
        base, hstr, model = key.split(":", 2)   # FT keys contain ":" themselves
        if hstr != f"h{horizon}" or "aggregate" not in val:
            continue
        meta, _tr, _va, _te = load_ds(base)
        m = {k: v for k, v in val["aggregate"].items() if not k.startswith("proto_")}
        rec = {"dataset": base, "model": model, "mode": "open",
               "domain": meta.domain.value, "task": "forecast",
               "frequency": meta.frequency.value, "horizon": horizon,
               "protocol": "spec-4.1-rolling", "score": float(m.get("score", 0.0)),
               "metrics": _native(m)}
        if (base, model, "spec-4.1-rolling") in existing:
            continue
        recs.append(rec)
        added += 1
    LEADERBOARD.write_text(json.dumps(recs, indent=1))
    print(f"[leaderboard] appended {added} spec-4.1 rows (h={horizon}); "
          f"total {len(recs)}", flush=True)
    return added


# --------------------------------------------------------------------------- #
args_ft: dict = {"device": None, "amp": None}


def _acquire_lock(force: bool = False) -> None:
    """One writer per GPU: two concurrent sweeps on a 6 GB card OOM each other."""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    if LOCK.exists() and not force:
        try:
            pid = int(LOCK.read_text().strip())
            os.kill(pid, 0)
        except (ValueError, ProcessLookupError, PermissionError):
            LOCK.unlink(missing_ok=True)          # stale lock
        else:
            raise SystemExit(
                f"another benchmark run is alive (pid {pid}); wait for it, or pass --force "
                f"if you are sure it is dead (lock: {LOCK})")
    LOCK.write_text(str(os.getpid()))


def main(argv=None) -> int:
    global FT_LR, FT_CONTEXT, FT_BS, HORIZONS, CONTEXTS
    ap = argparse.ArgumentParser(prog="tsfm4finance.bench.run")
    ap.add_argument("--stage", action="append", default=None,
                    choices=["tasks", "finetune", "forecast", "classify", "anomaly",
                             "trading", "valcheck", "derive", "leaderboard", "report"])
    ap.add_argument("--limit", type=int, default=None, help="limit number of series (debug)")
    ap.add_argument("--series", default=None,
                    help="comma-separated subset of series names (default: all 12)")
    ap.add_argument("--horizons", default=None, help="comma list, e.g. 1,5,12,21,63")
    ap.add_argument("--contexts", default=None, help="comma list of zero-shot contexts")
    ap.add_argument("--ft-bs", type=int, default=FT_BS, help="fine-tune batch size")
    ap.add_argument("--ft-amp", default="fp32", choices=["fp32", "bf16", "fp16"],
                    help="mixed precision for fine-tuning (bf16 halves activation memory)")
    ap.add_argument("--ft-lr", type=float, default=FT_LR)
    ap.add_argument("--ft-context", type=int, default=FT_CONTEXT,
                    help="fine-tune context length (also used by the FT forecast engines)")
    ap.add_argument("--ft-device", default=None, help="e.g. cuda:0 / cpu (fine-tune only)")
    ap.add_argument("--ft-epochs", type=int, default=None,
                    help="override the auto epoch count for fine-tuning")
    ap.add_argument("--force", action="store_true",
                    help="ignore the single-writer GPU lock (not recommended)")
    ap.add_argument("--trade-h", type=int, default=5)
    args = ap.parse_args(argv)
    stages = args.stage or ["tasks", "finetune", "forecast", "classify", "anomaly",
                            "trading", "valcheck", "derive", "leaderboard", "report"]
    FT_LR, FT_CONTEXT, FT_BS = args.ft_lr, args.ft_context, args.ft_bs
    if args.horizons:
        HORIZONS = [int(x) for x in args.horizons.split(",") if x.strip()]
    if args.contexts:
        CONTEXTS = [int(x) for x in args.contexts.split(",") if x.strip()]
    if args.series:
        want = {x.strip() for x in args.series.split(",") if x.strip()}
        T.REAL_SERIES = [(b, d) for b, d in T.REAL_SERIES if b in want]
        if not T.REAL_SERIES:
            print(f"no series match {args.series!r}"); return 2
    _acquire_lock(args.force)
    args_ft.update({"device": args.ft_device, "amp": None if args.ft_amp == "fp32" else args.ft_amp})
    res = load_results()
    for st in stages:
        t0 = time.time()
        print(f"\n=== stage {st} ===", flush=True)
        if st == "tasks":
            made = T.build_all()
            print(f"[tasks] {len(made)} real task datasets ready", flush=True)
        elif st == "finetune":
            stage_finetune(res, args.limit)
        elif st == "forecast":
            stage_forecast(res, args.limit)
        elif st == "classify":
            stage_classify(res, args.limit)
        elif st == "anomaly":
            stage_anomaly(res, args.limit)
        elif st == "trading":
            stage_trading(res, args.limit, h=args.trade_h)
        elif st == "valcheck":
            stage_valcheck(res, args.limit)
        elif st == "derive":
            stage_derive(res)
        elif st == "leaderboard":
            stage_leaderboard(res)
        elif st == "report":
            from tsfm4finance.bench import report
            report.build(res)
        print(f"=== stage {st} done in {time.time()-t0:.0f}s ===", flush=True)
    save_results(res)
    LOCK.unlink(missing_ok=True)
    print(f"\nresults -> {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
