"""Cell worker: executes one (env, stage, shard) slice of the benchmark grid.

One process per python environment, because the families cannot share one
interpreter (uni2ts pins torch<2.5, momentfm pins numpy==1.25, toto-2 needs
python>=3.12, TiRex-2 needs its own Triton build).  A worker:

  1. opens the shared SQLite cell store (WAL) and reclaims abandoned ``running``
     rows from a previous crash,
  2. selects the pending cells that belong to its env and shard,
  3. loads each model **once** and runs every cell that needs it,
  4. persists each cell as its own artifact + row, so an interrupt costs at most
     the cell that was in flight,
  5. walks a degradation ladder (amp -> shorter context -> smaller batch -> cpu)
     before giving up, recording the profile that actually worked.

Run through ``python -m tsfm4finance.bench.sweep`` (which spawns these); it can
also be invoked directly for debugging.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from collections import defaultdict
from pathlib import Path

import os
import numpy as np

from .config import load_grid
from .engines import build_engine
from .protocol import build_windows
from .scoring import score_forecast
from .state import CellStore

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- utils
def _vram_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return round(float(torch.cuda.memory_allocated()) / 1e6, 1)
    except Exception:                                          # pragma: no cover
        pass
    return 0.0


def ladder(amp: str, ctx: int, bs: int, device: str) -> list[tuple]:
    """Degradation ladder: keep the cell alive on smaller hardware if possible.

    Every step is recorded in the row's ``profile`` column, so a table can be
    rebuilt from *what actually ran* rather than from what was requested.
    """
    seen: set[tuple] = set()
    out: list[tuple] = []

    def add(step):
        if step not in seen:
            seen.add(step)
            out.append(step)

    add((amp, ctx, bs, device))
    if device.startswith("cuda"):
        add(("bf16", ctx, bs, device))
        add(("fp16", ctx, max(1, bs // 2), device))
    add((amp, max(64, ctx // 2), max(1, bs // 2), device))
    add((amp, max(64, ctx // 4), 1, device))
    add(("fp32", max(64, ctx // 4), 1, "cpu"))
    return out


class WindowCache:
    """Windows depend only on (dataset, horizon, span, ctx_cap): compute once."""

    def __init__(self, ctx_cap: int):
        self.ctx_cap = ctx_cap
        self.cache: dict[tuple, tuple] = {}

    def get(self, series: str, horizon: int, span: str):
        key = (series, int(horizon), span, self.ctx_cap)
        if key not in self.cache:
            from tsfm4finance.bench.download.common import load_dataset
            tr, va, te, meta = load_dataset(series)
            target = str(meta.get("target", "close"))
            wins, info = build_windows(tr, va, te, horizon=horizon, target=target,
                                       max_context=self.ctx_cap, span=span)
            if not wins:                       # too few origins: widen the span
                wins, info = build_windows(tr, va, te, horizon=horizon, target=target,
                                           max_context=self.ctx_cap, span="val+test")
                info["span_fallback"] = "val+test"
            full = np.concatenate([tr[target].to_numpy(float), va[target].to_numpy(float),
                                   te[target].to_numpy(float)])
            self.cache[key] = (wins, info, meta, tr[target].to_numpy(float), full)
        return self.cache[key]


def _engine_failed(store: CellStore, rows, err: str) -> None:
    """A model that cannot even be loaded is recorded, never silently dropped."""
    for r in rows:
        if store.claim(r["cell_id"]):
            store.skip(r["cell_id"], f"engine-unavailable: {err[:300]}")


# ---------------------------------------------------------------------- forecast
def run_forecast(store: CellStore, grid, rows, a) -> None:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        # an adapted checkpoint belongs to one series, so adapted cells are grouped
        # per (model, adaptation, series); zero-shot cells stay grouped per model
        key = (r["model"], r["adaptation"], r["series"] if r["adaptation"] != "zero" else "")
        groups[key].append(r)
    specs = grid.models()
    wc = WindowCache(a.ctx_cap)

    for (model, adaptation, _ser), grp in sorted(groups.items()):
        spec = specs.get(model)
        if spec is None:
            _engine_failed(store, grp, f"unknown model {model!r} in config")
            continue
        ckpt = a.ft_ckpt_for(model, adaptation, _ser) if adaptation != "zero" else None
        if adaptation != "zero" and not ckpt:
            _engine_failed(store, grp, f"no fine-tuned checkpoint for {model}/{adaptation} "
                                       f"(run: python -m tsfm4finance.bench.trainers "
                                       f"--model {model} --series <s> --budget {adaptation})")
            continue
        cache: dict[tuple, object] = {}

        def get_engine(step):
            """Engines are expensive (weights + compile); build one per profile."""
            if step not in cache:
                amp, ctx, bs, dev = step
                cache[step] = build_engine(spec, adaptation=adaptation, ft_ckpt=ckpt,
                                           device=dev, amp=amp, batch=bs, ctx_cap=ctx)
            return cache[step]

        try:                                        # fail fast if the model cannot load
            get_engine(ladder(a.amp, a.ctx_cap, a.batch, a.device)[0])
        except Exception as e:                                      # noqa: BLE001
            _engine_failed(store, grp, f"{type(e).__name__}: {e}")
            continue

        for r in grp:
            if not store.claim(r["cell_id"]):
                continue
            t0 = time.time()
            ok = False
            for step in ladder(a.amp, a.ctx_cap, a.batch, a.device):
                try:
                    eng = get_engine(step)
                    wins, info, meta, train_hist, _full = wc.get(r["series"], r["horizon"],
                                                                 r["span"] or "auto")
                    if not wins:
                        store.skip(r["cell_id"], "no-origins")
                        ok = True
                        break
                    pred = eng.predict([w.context for w in wins], int(r["horizon"]),
                                       panels=[w.panel for w in wins])
                    from .engines.baselines import NaiveEngine
                    naive = NaiveEngine().predict([w.context for w in wins],
                                                  int(r["horizon"])).point
                    cell = score_forecast(wins, pred, info=info, train_hist=train_hist,
                                          horizon=int(r["horizon"]), baseline_point=naive,
                                          model_key=r["model"], series=r["series"],
                                          adaptation=adaptation)
                    cell["engine"] = {"family": spec.family, "weights": spec.weights,
                                      "params_m": spec.params_m, "tier": spec.tier,
                                      "license": spec.license}
                    _ft = getattr(eng, "ft_load", None)
                    if _ft:
                        cell["engine"]["ft"] = {k: _ft[k] for k in
                                                ("checkpoint", "via", "missing", "unexpected")
                                                if k in _ft}
                    cell["profile"] = _profile(step)
                    store.finish(r["cell_id"], cell, profile=cell["profile"],
                                 seconds=round(time.time() - t0, 2), vram_mb=_vram_mb())
                    ok = True
                    break
                except Exception as e:                             # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"
                    if not _oom(err):
                        store.fail(r["cell_id"], err + "\n" + traceback.format_exc()[-800:],
                                   seconds=round(time.time() - t0, 2))
                        ok = True
                        break
                    store.note(r["cell_id"], f"degraded via {_profile(step)}")
            if not ok:
                store.skip(r["cell_id"], "degradation-ladder-exhausted (oom)")


def _profile(step) -> str:
    amp, ctx, bs, dev = step
    return f"{dev}/{amp}/ctx{ctx}/bs{bs}"


def _oom(msg: str) -> bool:
    m = msg.lower()
    return "out of memory" in m or "cuda oom" in m or "not enough memory" in m or "no available " in m


# ----------------------------------------------------------------------- trading
def run_trading(store: CellStore, grid, rows, a) -> None:
    """Signal-timing task (spec §2.3): forecast -> position -> net-of-cost PnL."""
    from .finance_metrics import (cost_slope, deflated_sharpe_ratio, market_capture,
                                  max_drawdown_duration, ulcer_index)
    from .trading import benchmark_trades, trade_eval, trading_returns

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["model"], r["adaptation"],
               a.base_series(r["series"]) if r["adaptation"] != "zero" else "")
        groups[key].append(r)
    specs = grid.models()
    wc = WindowCache(a.ctx_cap)

    for (model, adaptation, _ser), grp in sorted(groups.items()):
        spec = specs.get(model)
        if spec is None:
            _engine_failed(store, grp, f"unknown model {model!r}")
            continue
        ckpt = a.ft_ckpt_for(model, adaptation, _ser) if adaptation != "zero" else None
        if adaptation != "zero" and not ckpt:
            _engine_failed(store, grp, f"no fine-tuned checkpoint for {model}/{adaptation} "
                                       f"on {_ser} (train it: bench.trainers --model {model} "
                                       f"--series {_ser} --budget {adaptation})")
            continue
        engine = None
        for amp, ctx, bs, dev in ladder(a.amp, a.ctx_cap, a.batch, a.device):
            try:
                engine = build_engine(spec, adaptation=adaptation, ft_ckpt=ckpt, device=dev,
                                      amp=amp, batch=bs, ctx_cap=ctx)
                break
            except Exception as e:                                 # noqa: BLE001
                last = f"{type(e).__name__}: {e}"
        if engine is None:
            _engine_failed(store, grp, last)
            continue

        for r in grp:
            if not store.claim(r["cell_id"]):
                continue
            h = int(r["horizon"] or 5)
            t0 = time.time()
            try:
                base = r["series"].replace("_trade", "")
                wins, info, meta, _tr, full = wc.get(base, h, r["span"] or "auto")
                if len(wins) < 8:
                    store.skip(r["cell_id"], f"too-few-origins({len(wins)})")
                    continue
                pred = engine.predict([w.context for w in wins], h,
                                      panels=[w.panel for w in wins])
                qs = None
                if pred.quantiles is not None:
                    qs = [pred.quantiles[i] for i in range(len(wins))]
                origins = [w.origin for w in wins]
                fwd = np.array([float(np.log(float(w.truth[-1]) / float(w.context[-1])))
                                for w in wins])
                res = trade_eval([w.context for w in wins], pred.point, qs, origins,
                                 full, fwd, h, cost_grid=list(a.costs))
                bench = benchmark_trades(full, origins, fwd, h, cost_grid=list(a.costs))
                cell = {"stage": "trading", "series": r["series"], "model": model,
                        "adaptation": adaptation, "horizon": h, "span": info.get("span"),
                        "n_origins": len(wins), "no_leakage": info.get("no_leakage"),
                        "rules": res, "benchmarks": bench,
                        "profile": getattr(engine, "device", a.device)}
                # DSR / capture / ulcer, computed on the vol-target net returns
                from .trading import positions, signal_from_forecast
                sg = [signal_from_forecast(w.context, pred.point[i],
                                           qs[i] if qs else None, h) for i, w in enumerate(wins)]
                posn = positions(np.array([x[0] for x in sg]), np.array([x[1] for x in sg]),
                                 "voltarget")
                cb_mid = sorted(a.costs)[len(a.costs) // 2]
                net = trading_returns(posn, fwd, cost_bps=cb_mid)
                cell["extras"] = {
                    "ulcer_index": ulcer_index(net),
                    "dd_duration_y": max_drawdown_duration(net),
                    "capture": market_capture(posn * fwd, fwd),
                    "dsr": deflated_sharpe_ratio(net, n_trials=int(a.n_trials or 20)),
                    "cost_slope": cost_slope({cb: res["voltarget"][f"{cb:.0f}bps"]["sharpe_net"]
                                             for cb in a.costs}),
                }
                store.finish(r["cell_id"], cell, profile=cell["profile"],
                             seconds=round(time.time() - t0, 2), vram_mb=_vram_mb())
            except Exception as e:                                 # noqa: BLE001
                store.fail(r["cell_id"], f"{type(e).__name__}: {e}\n"
                                         + traceback.format_exc()[-800:],
                           seconds=round(time.time() - t0, 2))


# -------------------------------------------------------------------- classify
def run_classify(store: CellStore, grid, rows, a) -> None:
    """Discriminative task families (spec §2.2).

    Routes, in order of what the model offers:
      * ``native_head``       - models with a pretrained classification head (MOMENT)
      * ``embedding_probe``   - linear/GBM probe on encoder embeddings
      * ``forecast_features`` - probe on the model's own forecast + quantiles
      * ``stat``              - statistical features only (the honest baseline)
    """
    from .probe2 import fit_eval, stat_features

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["model"], r["adaptation"],
               a.base_series(r["series"]) if r["adaptation"] != "zero" else "")
        groups[key].append(r)
    specs = grid.models()
    wc = WindowCache(a.ctx_cap)

    for (model, adaptation, _ser), grp in sorted(groups.items()):
        spec = specs.get(model)
        if spec is None:
            _engine_failed(store, grp, f"unknown model {model!r}")
            continue
        ckpt = a.ft_ckpt_for(model, adaptation, _ser) if adaptation != "zero" else None
        if adaptation != "zero" and not ckpt:
            _engine_failed(store, grp, f"no fine-tuned checkpoint for {model}/{adaptation} "
                                       f"on {_ser} (train it: bench.trainers --model {model} "
                                       f"--series {_ser} --budget {adaptation})")
            continue
        engine = None
        if model != "stat":
            for amp, ctx, bs, dev in ladder(a.amp, a.ctx_cap, a.batch, a.device):
                try:
                    engine = build_engine(spec, adaptation=adaptation, ft_ckpt=ckpt,
                                          device=dev, amp=amp, batch=bs, ctx_cap=ctx)
                    break
                except Exception as e:                             # noqa: BLE001
                    last = f"{type(e).__name__}: {e}"
            if engine is None:
                _engine_failed(store, grp, last)
                continue

        for r in grp:
            if not store.claim(r["cell_id"]):
                continue
            route = str(r["span"] or "probe:stat").split(":")[-1]
            t0 = time.time()
            try:
                h = int(r["horizon"] or 1)
                wins, labels, info = _supervised_windows(store, r["series"], a.ctx_cap)
                if labels is None or len(wins) < 5:
                    store.skip(r["cell_id"], "no-label-column-or-too-few-windows")
                    continue
                if engine is not None and route == "embedding_probe" and not spec.can("embed"):
                    store.skip(r["cell_id"], "no-embeddings-capability")
                    continue
                if engine is not None and route == "native_head" and not spec.can("clf_head"):
                    store.skip(r["cell_id"], "no-native-clf-head")
                    continue
                if route == "stat" or model == "stat":
                    feats = np.array([stat_features(w.context) for w in wins])
                elif route == "embedding_probe":
                    emb = engine.encode([w.context for w in wins])
                    feats = np.asarray(emb, dtype=float)
                elif route == "native_head":
                    proba = engine.classify([w.context for w in wins], n_classes=len(labels))
                    cell = fit_eval(None, wins, proba, labels, route=route)
                    cell.update({"stage": "classify", "series": r["series"], "model": model,
                                 "adaptation": adaptation, "route": route,
                                 "n_origins": len(wins)})
                    store.finish(r["cell_id"], cell, seconds=round(time.time() - t0, 2),
                                 vram_mb=_vram_mb())
                    continue
                else:                                    # forecast_features
                    pred = engine.predict([w.context for w in wins], max(1, h // 5 or 1))
                    feats = np.array([np.r_[stat_features(w.context), p, _band(q)]
                                      for w, p, q in zip(wins, pred.point,
                                                        _q_iter(pred, len(wins)))])
                cell = fit_eval(feats, wins, None, labels, route=route)
                cell.update({"stage": "classify", "series": r["series"], "model": model,
                             "adaptation": adaptation, "route": route,
                             "n_origins": len(wins), "no_leakage": info.get("no_leakage")})
                store.finish(r["cell_id"], cell, seconds=round(time.time() - t0, 2),
                             vram_mb=_vram_mb())
            except Exception as e:                                 # noqa: BLE001
                store.fail(r["cell_id"], f"{type(e).__name__}: {e}\n"
                                         + traceback.format_exc()[-800:],
                           seconds=round(time.time() - t0, 2))


def _q_iter(pred, n):
    if pred.quantiles is None:
        return [None] * n
    return [pred.quantiles[i] for i in range(n)]


def _band(q):
    if q is None:
        return np.zeros(2)
    return np.array([float(q[-1, 8] - q[-1, 0]), float(np.mean(q[:, 4] - q[:, 0]))])


def _supervised_windows(store, name: str, ctx_cap: int):
    """Task datasets (<base>_dir5/_volreg/_extreme) carry a ``label`` column."""
    from tsfm4finance.bench.download.common import load_dataset
    tr, va, te, meta = load_dataset(name)
    import pandas as pd
    wins, info = build_windows(tr, va, te, horizon=1, target="close",
                               max_context=ctx_cap, span="val+test")
    full = pd.concat([tr, va, te], ignore_index=True)
    lab = full["label"].to_numpy(dtype=float) if "label" in full.columns else None
    if lab is None:
        return wins, None, info
    keep = [i for i, w in enumerate(wins)
            if np.isfinite(lab[w.origin]) and 0 <= w.origin < len(lab)]
    wins = [wins[i] for i in keep]
    return wins, np.array([lab[w.origin] for w in wins]), info


# ----------------------------------------------------------------------- anomaly
def run_anomaly(store: CellStore, grid, rows, a) -> None:
    from .probe2 import anomaly_eval, zscore_scores

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["model"], r["adaptation"],
               a.base_series(r["series"]) if r["adaptation"] != "zero" else "")
        groups[key].append(r)
    specs = grid.models()
    wc = WindowCache(a.ctx_cap)
    for (model, adaptation, _ser), grp in sorted(groups.items()):
        spec = specs.get(model)
        if spec is None:
            _engine_failed(store, grp, f"unknown model {model!r}")
            continue
        ckpt = a.ft_ckpt_for(model, adaptation, _ser) if adaptation != "zero" else None
        if adaptation != "zero" and not ckpt:
            _engine_failed(store, grp, f"no fine-tuned checkpoint for {model}/{adaptation} "
                                       f"on {_ser}")
            continue
        route = grp[0]["span"] or "zscore"
        engine = None
        if route != "zscore":
            try:
                engine = build_engine(spec, adaptation=adaptation, ft_ckpt=ckpt,
                                      device=a.device, amp=a.amp, batch=a.batch,
                                      ctx_cap=a.ctx_cap)
            except Exception as e:                                 # noqa: BLE001
                _engine_failed(store, grp, f"{type(e).__name__}: {e}")
                continue
            if route == "band_violation" and not spec.can("quantiles"):
                _engine_failed(store, grp, "model has no quantile output")
                continue
            if route == "native_reconstruction" and not spec.can("anomaly_head"):
                _engine_failed(store, grp, "model has no anomaly/reconstruction head")
                continue
        for r in grp:
            if not store.claim(r["cell_id"]):
                continue
            t0 = time.time()
            try:
                wins, labels, info = _supervised_windows(store, r["series"], a.ctx_cap)
                if labels is None or len(wins) < 5:
                    store.skip(r["cell_id"], "no-label-column-or-too-few-windows")
                    continue
                if route == "zscore" or engine is None:
                    scores = zscore_scores([w.context for w in wins])
                elif route == "band_violation":
                    pred = engine.predict([w.context for w in wins], 1)
                    scores = _band_violation(wins, pred)
                else:
                    scores = engine.anomaly_scores([w.context for w in wins])
                cell = anomaly_eval(scores, labels)
                cell.update({"stage": "anomaly", "series": r["series"], "model": model,
                             "adaptation": adaptation, "route": route,
                             "n_origins": len(wins), "no_leakage": info.get("no_leakage")})
                store.finish(r["cell_id"], cell, seconds=round(time.time() - t0, 2),
                             vram_mb=_vram_mb())
            except Exception as e:                                 # noqa: BLE001
                store.fail(r["cell_id"], f"{type(e).__name__}: {e}\n"
                                         + traceback.format_exc()[-800:],
                           seconds=round(time.time() - t0, 2))


def _band_violation(wins, pred) -> np.ndarray:
    """1 if the realised next move falls outside the model's own 10-90% band."""
    out = []
    for i, w in enumerate(wins):
        q = pred.quantiles[i] if pred.quantiles is not None else None
        if q is None:
            out.append(0.0)
            continue
        lo, hi = float(np.min(q[0])), float(np.max(q[-1]))
        out.append(float(not (lo <= float(w.truth[0]) <= hi)))
    return np.array(out)


STAGES = {"forecast": run_forecast, "trading": run_trading,
          "classify": run_classify, "anomaly": run_anomaly}


# ------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    ap.add_argument("--env", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--config", default=None)
    ap.add_argument("--db", default=str(REPO / "reports" / "state.sqlite3"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", default="fp32")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ctx-cap", type=int, default=512)
    ap.add_argument("--costs", default="0,10,30")
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--max-cells", type=int, default=0)
    ap.add_argument("--ft-dir", default=str(REPO / "reports" / "ft_ckpts"))
    ap.add_argument("--force", action="store_true",
                    help="re-queue terminal cells of this stage (done/failed/skipped)")
    ap.add_argument("--weights-root", default=os.environ.get("TSFM_WEIGHTS_ROOT", ""),
                    help="directory holding git-cloned / hf-downloaded models")
    a = ap.parse_args()
    if a.weights_root:
        os.environ["TSFM_WEIGHTS_ROOT"] = a.weights_root

    a.costs = [float(x) for x in str(a.costs).split(",") if x.strip()]

    def ft_ckpt_for(model: str, adaptation: str, series: str = "") -> str | None:
        """Trainers write ``reports/ft_ckpts/<model>__<series>__<budget>.pt``.

        The checkpoint must belong to *this* cell's series: an adapted cell scored on
        series B with a checkpoint trained on series A would be leakage, so there is
        deliberately no cross-series fallback.
        """
        d = Path(a.ft_dir)
        if not d.exists():
            return None
        pats = [f"{model}__{series}__{adaptation}.pt", f"{model}__{series}__{adaptation}"]
        if not series:
            pats = [f"{model}__*__{adaptation}.pt", f"{model}/*__{adaptation}.pt"]
        for pat in pats:
            hits = sorted(d.glob(pat)) or sorted((d / model).glob(pat.split("/")[-1]))
            for h in hits:
                if h.is_dir() and (any(h.glob("*.json")) or any(h.glob("*.safetensors"))):
                    return str(h)
                if h.is_file():
                    return str(h)
        return None

    a.ft_ckpt_for = ft_ckpt_for                       # type: ignore[attr-defined]

    def base_series(name: str) -> str:
        """Derived task datasets (``*_trade``/``*_dir5``/``*_volreg``/``*_extreme``)
        are scored on the same underlying series the trainer checkpoint was fit on."""
        for suf in ("_trade", "_dir5", "_volreg", "_extreme"):
            if name.endswith(suf):
                return name[: -len(suf)]
        return name

    a.base_series = base_series                        # type: ignore[attr-defined]

    grid = load_grid(a.config)
    store = CellStore(a.db)
    store.reset_running()
    if a.force:
        store.retry_failed(stages=(a.stage,))
        for r in store.rows(stage=a.stage, status="done"):
            store.con.execute("UPDATE cells SET status='pending' WHERE cell_id=?",
                              (r["cell_id"],))
        store.con.commit()
    rows = [r for r in store.rows(stage=a.stage, status="pending")
            if grid.models().get(r["model"]) is not None
            and grid.models()[r["model"]].env == a.env]
    if a.n_shards > 1:
        import hashlib
        rows = [r for r in rows
                if int(hashlib.sha256(r["cell_id"].encode()).hexdigest()[:8], 16) % a.n_shards
                == a.shard]
    if a.max_cells:
        rows = rows[:a.max_cells]
    print(json.dumps({"stage": a.stage, "env": a.env, "shard": f"{a.shard}/{a.n_shards}",
                      "cells": len(rows), "grid": grid.digest()}))
    if rows:
        STAGES[a.stage](store, grid, rows, a)
    print(json.dumps({"counts": store.counts().get(a.stage, {})}))


if __name__ == "__main__":
    main()
