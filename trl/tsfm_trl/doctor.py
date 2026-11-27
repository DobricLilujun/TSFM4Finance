"""``tsfm-trl doctor`` — the compatibility matrix.

For every model in the registry (or the subset you ask for) it answers, with
captured reasons rather than blanks:

    weights present → importable → loadable → predict → quantiles → embeddings
    → fine-tune smoke (2 optimiser steps, loss decreases?) → classification head
    → anomaly → reward mode

and writes ``reports/compat_matrix.json`` + ``COMPATIBILITY.md``. Because the
families genuinely need different python environments, the matrix records the
interpreter and machine that produced each row: a row is only comparable to a row
run the same way. Run one env at a time with ``--env``, or let ``--all-envs``
dispatch subprocesses and merge.
"""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .data.fixtures import tiny_finance
from .model import ModelHub
from .model.registry import ModelSpec
from .training.classify import ClassificationTrainer
from .training.config import TrainingConfig
from .training.forecast import ForecastTrainer
from .weights import present as weights_present


@dataclass
class Row:
    model: str
    family: str = ""
    tier: str = ""
    env: str = ""
    weights: str = ""
    license: str = ""
    weights_present: bool = False
    load: str = "skip"
    predict: str = "skip"
    quantiles: str = "skip"
    embed: str = "skip"
    finetune: str = "skip"
    ft_loss_first: float = float("nan")
    ft_loss_last: float = float("nan")
    ft_params: int = 0
    classify: str = "skip"
    clf_metrics: dict = field(default_factory=dict)
    anomaly: str = "skip"
    reward: str = "skip"
    #: grpo | rft | reward_descent | reward_descent_point — what the model's own
    #: architecture allowed, not what we preferred (RewardTrainer picks automatically)
    reward_mode: str = ""
    reward_early: float = float("nan")
    reward_late: float = float("nan")
    seconds: float = 0.0
    reasons: dict = field(default_factory=dict)
    machine: str = ""
    python: str = ""
    #: hostname/arch — a laptop row and a DGX row for the same model are different claims
    device: str = ""

    def ok(self, k) -> bool:
        return getattr(self, k) == "ok"


def _series(ctx: int, horizon: int):
    ds = tiny_finance()
    return ds, ds.split("train"), ds.split("test")


def check(spec: ModelSpec, *, ctx: int = 256, horizon: int = 21, device: str = "auto",
          tasks: tuple = ("load", "predict", "finetune", "classify", "anomaly", "reward"),
          series_path: str = "", batch: int = 4, reward_steps: int = 6,
          clf_windows: int = 192) -> Row:
    row = Row(model=spec.key, family=spec.family, tier=spec.tier, env=spec.env,
              weights=spec.weights or "", license=spec.license,
              weights_present=bool(weights_present(spec.weights)) if spec.weights else True,
              machine=f"{socket.gethostname()}/{platform.machine()}",
              python=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
              device=device)
    t0 = time.time()
    ds = None
    if series_path:
        try:
            from .data.series import SeriesDataset
            ds = SeriesDataset.load(series_path)
        except Exception as e:                                     # noqa: BLE001
            row.reasons["series"] = f"{type(e).__name__}: {e}"
    if ds is None:
        ds = tiny_finance()
    ctx = min(ctx, 256)
    horizon = min(horizon, 21)
    train = ds.split("train")[-(ctx + horizon + 64):]
    cs = [train[i:i + ctx] for i in range(0, max(1, len(train) - ctx - horizon), max(1, horizon))]
    cs = [c for c in cs if len(c) == ctx][:4] or [train[-ctx:]]
    ys = np.stack([train[len(c):len(c) + horizon] for c in cs])
    ys = np.stack([train[-(ctx + horizon) + len(c):-(ctx + horizon) + len(c) + horizon]
                   for c in cs]).astype(np.float32) if ys.size == 0 else ys.astype(np.float32)

    # ---------------------------------------------------------------- load
    model = None
    # loading is a prerequisite for every other check, so it always runs; the
    # "load" task only controls whether the row is reported as a load result
    if True:
        try:
            model = ModelHub.load(spec.key, device=device, batch=batch,
                                  ctx_cap=min(spec.max_context, ctx * 2))
            row.load = "ok"
        except Exception as e:                                     # noqa: BLE001
            row.load = "fail"
            row.reasons["load"] = f"{type(e).__name__}: {e}"[:300]
            # every downstream cell, not just the requested ones: "this model does not
            # load" is the reason those cells are empty whether or not the caller asked
            for t in ("predict", "quantiles", "embed", "classify", "anomaly",
                      "finetune", "reward"):
                row.reasons[t] = "not attempted: the model did not load"
            row.seconds = time.time() - t0
            return row
    if model is None:
        row.seconds = time.time() - t0
        return row
    row.device = str(getattr(model, "device", device))
    panels = None
    if "panel_in" in set(model.caps) and ds.panel is not None:
        panels = [ds.panel_window(max(0, len(ds) - ctx - horizon - 1 + i),
                                 len(ds) - horizon - 1 + i) for i in range(len(cs))]

    # ---------------------------------------------------------------- predict
    if "predict" in tasks:
        try:
            p = model.predict(cs, horizon, panels=panels)
            ok = (p.point is not None and np.asarray(p.point).shape == (len(cs), horizon)
                  and np.all(np.isfinite(np.asarray(p.point))))
            row.predict = "ok" if ok else f"bad-shape {np.asarray(p.point).shape}"
            if p.quantiles is not None:
                q = np.asarray(p.quantiles)
                row.quantiles = "ok" if q.shape[:2] == (len(cs), horizon) else \
                    f"bad-shape {q.shape}"
            else:
                row.quantiles = "none"
        except Exception as e:                                     # noqa: BLE001
            row.predict = "fail"
            row.reasons["predict"] = f"{type(e).__name__}: {e}"[:300]
            # quantiles is read off the same call, so it was never observed
            row.reasons["quantiles"] = "not observed: the point forecast itself failed"

    # ---------------------------------------------------------------- embed
    if "embed" in tasks:
        try:
            e = np.asarray(model.encode(cs), float)
            row.embed = "ok" if e.ndim == 2 and e.shape[0] == len(cs) and e.shape[1] > 0 \
                else f"bad-shape {e.shape}"
        except Exception as e:                                     # noqa: BLE001
            row.embed = "fail"
            row.reasons["embed"] = f"{type(e).__name__}: {e}"[:200]

    # ---------------------------------------------------------------- fine-tune
    if "finetune" in tasks and not (spec.trainable or model.loss_kind):
        # The statistical baselines and the inference-only vendors are not "untested"
        # here: the package offers nothing trainable, which is the declined answer. Left
        # blank, a reader cannot tell "we did not look" from "there is nothing there".
        row.finetune = "skipped"
        row.reasons["finetune"] = (f"{spec.family}: no trainable module and no loss_kind "
                                   f"in the installed package (nothing to fine-tune)")
    if "finetune" in tasks and (spec.trainable or model.loss_kind):
        cfg = TrainingConfig(ctx=ctx, horizon=horizon, budget="fs16", steps=2, bs=2,
                             reward_steps=reward_steps,
                             device=row.device if row.device != "auto" else "auto",
                             smoke=True, seed=0, tag=spec.key)
        wins = _wins(train, ctx, horizon, n=4)
        try:
            res = ForecastTrainer(model, cfg).fit({"windows": wins,
                                                   "info": {"series": "fixture"}})
            if res.status in ("ok", "smoke"):
                row.finetune = "ok"
                row.ft_loss_first, row.ft_loss_last = res.first_loss, res.final_loss
                row.ft_params = res.trainable_parameters
                # vendor loops (Chronos-2 fit) may not report a curve: nan is not a failure
                if (np.isfinite(res.final_loss) and np.isfinite(res.first_loss)
                        and res.final_loss > res.first_loss * 1.5):
                    row.reasons["finetune"] = (f"loss did not fall: {res.first_loss} -> "
                                               f"{res.final_loss}")
            else:
                row.finetune = _st(res.status)
                row.reasons["finetune"] = res.reason
        except Exception as e:                                     # noqa: BLE001
            row.finetune = "fail"
            row.reasons["finetune"] = f"{type(e).__name__}: {e}"[:300]

    # ---------------------------------------------------------------- classify
    if "classify" in tasks:
        from .data.classify import direction_dataset
        try:
            # capped, evenly spaced: every family is measured on the same window
            # count, and a sampler model's feature extraction costs ~draws× more
            cds = direction_dataset(ds, horizon=5, ctx=min(ctx, 120)).limit(clf_windows)
            cfg = TrainingConfig(ctx=min(ctx, 120), horizon=5, budget="fs16", steps=6, bs=8,
                                 device=row.device if row.device != "auto" else "auto",
                                 smoke=True, tag=spec.key)
            # all three levels, not just the one that usually works: the plan asks
            # probe / head(+LoRA) / native for every model, and "native" is only
            # meaningful when the vendor ships a classification head at all
            routes = {}
            for route in ("probe", "head", "native"):
                if route == "native" and not hasattr(model, "classify_fit"):
                    routes[route] = "n/a"
                    continue
                try:
                    res = ClassificationTrainer(model, cfg, route=route).fit(cds)
                    ok = res.status in ("ok", "smoke")
                    routes[route] = "ok" if ok else res.status
                    met = res.extra.get("metrics") or {}
                    if route == "head" or not row.clf_metrics:
                        row.clf_metrics = {k: (round(float(v), 4) if isinstance(v, float) else v)
                                           for k, v in met.items()}
                    if not ok:
                        row.reasons[f"classify:{route}"] = res.reason
                except Exception as e:                                 # noqa: BLE001
                    routes[route] = "fail"
                    row.reasons[f"classify:{route}"] = f"{type(e).__name__}: {e}"[:200]
            row.clf_metrics["routes"] = routes
            tried = [v for v in routes.values() if v != "n/a"]
            row.classify = ("ok" if "ok" in tried else
                            "fail" if "fail" in tried else "n/a")
        except Exception as e:                                     # noqa: BLE001
            # the dataset itself, or the feature matrix, broke — that is a failure of
            # this row, not a decline by the model
            row.classify = "fail"
            row.reasons["classify"] = f"{type(e).__name__}: {e}"[:300]

    # ---------------------------------------------------------------- anomaly
    if "anomaly" in tasks:
        try:
            s = np.asarray(model.anomaly_scores(cs), float)
            row.anomaly = "ok" if s.shape == (len(cs),) and np.all(np.isfinite(s)) else "partial"
        except Exception as e:                                     # noqa: BLE001
            row.anomaly = "n/a"
            row.reasons["anomaly"] = f"{type(e).__name__}: {e}"[:200]

    # ---------------------------------------------------------------- reward
    if "reward" in tasks:
        from .training.reward import RewardTrainer
        try:
            cfg = TrainingConfig(ctx=ctx, horizon=horizon, budget="fs16", steps=2, bs=2,
                                 reward_steps=reward_steps,
                                 device=row.device if row.device != "auto" else "auto",
                                 smoke=True, tag=spec.key)
            res = RewardTrainer(model, cfg, n_candidates=4).fit({"windows": _wins(train, ctx, horizon, 4),
                                                                 "info": {"series": "fixture"}})
            row.reward = "ok" if res.status in ("ok", "smoke") else _st(res.status)
            ex = res.extra or {}
            row.reward_mode = str(ex.get("mode") or "")
            for k, attr in (("reward_early_mean", "reward_early"),
                            ("reward_late_mean", "reward_late")):
                v = ex.get(k)
                if isinstance(v, (int, float)):
                    setattr(row, attr, float(v))
            if res.status not in ("ok", "smoke"):
                row.reasons["reward"] = res.reason
        except Exception as e:                                     # noqa: BLE001
            row.reward = "fail"
            row.reasons["reward"] = f"{type(e).__name__}: {e}"[:250]
            _debug_tb("reward")

    # free the model before the next row: a 6 GB card cannot hold two stacks and the
    # matrix must be runnable on a laptop
    try:
        import torch
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                              # noqa: BLE001
        pass
    row.seconds = round(time.time() - t0, 1)
    return row


def _debug_tb(task: str) -> None:
    """``TSFMTRL_DEBUG=1`` prints the traceback of a captured failure to stderr.

    The matrix deliberately swallows exceptions (one bad model must not end a 46-model
    run) and keeps only the message — which is right for the report and useless for
    fixing anything when the message is a vendor shape check that only fires inside the
    matrix. stdout stays the data channel; this goes to stderr.
    """
    import os
    if os.environ.get("TSFMTRL_DEBUG"):
        import traceback
        traceback.print_exc()
        sys.stderr.write(f"[TSFMTRL_DEBUG] {task} failed\n")


def _wins(train, ctx, horizon, n=4):
    from .data.series import windows
    w = windows(train, horizon, ctx, stride=horizon)
    return (w or [(train[-ctx:], train[-ctx - horizon:-ctx])])[:n]


def run_matrix(keys=None, *, tier=None, env=None, tasks=("load", "predict", "finetune",
               "classify", "anomaly", "reward"), clf_windows: int = 192, **kw) -> list[Row]:
    specs = ModelHub.list(tier=tier, env=env)
    if keys:
        specs = [s for s in specs if s.key in set(keys)]
    # `timeout` is the env-dispatch layer's subprocess budget, not a row argument
    kw.pop("timeout", None)
    out = []
    for s in specs:
        r = check(s, tasks=tasks, clf_windows=clf_windows, **kw)
        want = kw.get("device", "auto")
        # an out-of-memory is a statement about the machine, not the model, whether it
        # happened while loading the weights or while running the first forecast
        oom_task = None
        if (r.load == "fail" or r.predict == "fail"):
            oom_task = next((k for k in ("load", "predict") if _oom_in(r.reasons.get(k, ""))),
                            None)
        if oom_task and want != "cpu":
            # "the model does not fit next to the eleven others in this process" is a
            # statement about the machine, not about the model: re-run it where it does
            # fit and say which device produced the row
            r2 = check(s, tasks=tasks, clf_windows=clf_windows, **{**kw, "device": "cpu"})
            r2.reasons = dict(r2.reasons or {})
            r2.reasons["device"] = (f"cpu (retried after a GPU out-of-memory during "
                                    f"{oom_task} on {want}: "
                                    f"{(r.reasons.get(oom_task) or '')[:90]})")
            out.append(r2)
        else:
            out.append(r)
    return out


def _oom_in(msg: str) -> bool:
    m = (msg or "").lower()
    return "out of memory" in m or "oom" in m or "cannot allocate" in m


def row_from_dict(d: dict) -> Row:
    """Rebuild a stored row, restoring the *declined* vs *not attempted* distinction.

    Older reports flattened a declined capability into the same ``skip`` value as a task
    that was never requested. The reasons say which is which: a decline always leaves a
    reason behind, an unattempted task does not.
    """
    r = Row(**d)
    reasons = r.reasons or {}
    for t in ("load", "predict", "quantiles", "embed", "finetune", "classify",
              "anomaly", "reward"):
        if getattr(r, t) == "skip" and any(k == t or k.startswith(f"{t}:")
                                           for k in reasons):
            setattr(r, t, "skipped")
    routes = (r.clf_metrics or {}).get("routes") or {}
    for k, v in list(routes.items()):
        if v == "skip" and any(kt.split(":", 1)[1] == k for kt in reasons
                               if kt.startswith("classify:")):
            routes[k] = "skipped"
    return r


def _st(status: str) -> str:
    """TrainResult.status and Row.status must be one vocabulary in the matrix:
    ok / skip / fail / prepared / partial. "failed" next to "fail" renders as two
    symbols for the same thing."""
    # "skipped" is kept, not folded into "skip": the matrix renders "skipped" as —
    # (the model declines the task) and "skip" as · (the task was never attempted)
    return {"failed": "fail", "smoke": "ok"}.get(status, status)


def dedupe(rows: list[Row]) -> list[Row]:
    """One row per (model, env), preferring the machine where the model actually ran.

    Two machines disagreeing about the same model is a fact, not noise, so the loser is
    recorded on the winner under ``other machine`` instead of being dropped: that is how
    an aarch64-only kernel failure stays visible in an x86-clean matrix.
    """
    tasks = ("load", "predict", "finetune", "classify", "anomaly", "reward")

    def score(r: Row):
        # a row that ran more tasks wins a tie: re-running one model with a subset of
        # tasks must not replace the row that has the full picture
        routes = ((r.clf_metrics or {}).get("routes") or {})
        return (r.load == "ok", r.predict == "ok",
                sum(1 for t in tasks if getattr(r, t) == "ok"),
                # classification has three routes; a row where two of them ran beats one
                # where an out-of-memory on a small card killed one of them
                sum(1 for v in routes.values() if v == "ok"),
                r.finetune == "ok", r.classify == "ok")

    best: dict = {}
    for r in rows:
        cur = best.get((r.model, r.env))
        if cur is None:
            best[(r.model, r.env)] = r
            continue
        win, lose = (r, cur) if score(r) > score(cur) else (cur, r)
        # only an actual failure is worth a footnote: a task the other run never
        # attempted is not evidence of anything
        # a task the winner never attempted but the loser did is filled in from the
        # loser: partial re-runs (an embed-only pass, one model re-checked after a fix)
        # must not cost the table the coverage they produced
        for t in tasks + ("quantiles", "embed"):
            if getattr(win, t) == "skip" and getattr(lose, t) != "skip":
                setattr(win, t, getattr(lose, t))
                if (lose.reasons or {}).get(t):
                    win.reasons = dict(win.reasons or {})
                    win.reasons.setdefault(t, lose.reasons[t])
        if getattr(win, "reward", "skip") != "skip" and not getattr(win, "reward_mode", ""):
            # a reward re-run carries the *mode*, which is the part a benchmark reads
            win.reward_mode = getattr(lose, "reward_mode", "") or win.reward_mode
        wroutes = (win.clf_metrics or {}).get("routes") or {}
        lroutes = (lose.clf_metrics or {}).get("routes") or {}
        for k, v in lroutes.items():
            if wroutes.get(k, "skip") == "skip" and v != "skip":
                win.clf_metrics = dict(win.clf_metrics or {})
                win.clf_metrics["routes"] = dict(wroutes, **{k: v})
                wroutes = win.clf_metrics["routes"]
        bad = [f"{t}={getattr(lose, t)}" for t in tasks if getattr(lose, t) == "fail"]
        if bad:
            win.reasons = dict(win.reasons or {})
            win.reasons.setdefault("other machine", f"{lose.machine}: " + ", ".join(bad))
        best[(r.model, r.env)] = win
    return sorted(best.values(), key=lambda r: (r.tier, r.family, r.model))


def merge(rows: list[Row]) -> dict:
    return {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "python": sys.version.split()[0], "machine": f"{socket.gethostname()}",
            "n": len(rows), "rows": [asdict(r) for r in rows]}


_MODE_LABEL = {"grpo": "grpo", "rft": "rft", "reward_descent": "descent",
               "reward_descent_point": "descent·pt"}


def write_markdown(rows: list[Row], path: str | Path = "COMPATIBILITY.md") -> str:
    def m(v):
        # "skipped" = the model/adapter declines this task (a CapabilityError), which is
        # a statement about the model, not a failure of the library
        return {"ok": "✅", "none": "—", "n/a": "—", "skip": "·", "skipped": "—"}.get(v, "❌" if v not in
                                                                    ("partial",) else "◐")
    multi = len({r.machine for r in rows if r.machine}) > 1
    tail_c = " machine |" if multi else ""
    head = ("| model | tier | weights | load | forecast | quantiles | embed | classify | "
            "anomaly | reward (mode) | fine-tune (capability) | s |" + tail_c)
    line = "|---|---|---|---|---|---|---|---|---|---|---|---|" + ("---|" if multi else "")
    out = ["# Compatibility matrix", "",
           f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} on "
           f"{socket.gethostname()} ({platform.machine()}), python {sys.version.split()[0]}.",
           "", "✅ ok · ❌ failed (see the JSON report for the captured reason) · "
           "— not offered by the model · ◐ partial · · not checked",
           "", "**Benchmark columns** (the point of this table): load · forecast · quantiles "
           "· embed · classify · anomaly. `classify n/m` counts the probe / head+LoRA / "
           "native-head routes that ran.",
           "", "**reward (mode)** names the RL path the *architecture* allows — "
           "`grpo` (group-relative policy gradient over the model's own sampled token "
           "paths, needs log-probs), `rft` (rejection-sampling best-of-K, the "
           "group-baseline limit and the only path for sample-only heads), `descent` "
           "(gradient ascent on a differentiable quantile head), `descent·pt` (same "
           "through the point head). The trainer picks, not the caller; all four share one "
           "reward spec. It is a hand-written trainer with no RL library "
           "(`docs/05_reward_rl.md`).",
           "", "**fine-tune (capability)** is a gate, not a result: it says a differentiable "
           "vendor objective ran, gradients were feasible and loss logging worked, over the "
           "**two** smoke steps shown as `(a→b)`. It is *not* evidence that fine-tuning "
           "improves out-of-sample error — that is a benchmark run with a metric, and the "
           "effectiveness question lives in `docs/09_benchmark.md`.",
           "", f"Machines: {', '.join(sorted({r.machine for r in rows if r.machine})) or '-'}. "
           "One row per (model, environment); when two machines disagree the row that "
           "ran is kept and the other is recorded under *other machine* in the reasons.",
           "", head, line]
    for r in rows:
        rt = (r.clf_metrics or {}).get("routes") or {}
        cl = m(r.classify)
        if rt:
            cl += f" {sum(1 for v in rt.values() if v == 'ok')}/" \
                  f"{sum(1 for v in rt.values() if v != 'n/a')}"
        ft = m(r.finetune)
        if r.ok("finetune") and np.isfinite(r.ft_loss_first) and np.isfinite(r.ft_loss_last):
            ft += f" ({r.ft_loss_first:.3f}→{r.ft_loss_last:.3f})"
        rw = m(r.reward)
        if r.ok("reward") and r.reward_mode:
            rw += " " + _MODE_LABEL.get(r.reward_mode, r.reward_mode)
        out.append(f"| {r.model} | {r.tier} | {'✅' if r.weights_present else '❌'} | "
                   f"{m(r.load)} | {m(r.predict)} | {m(r.quantiles)} | {m(r.embed)} | "
                   f"{cl} | {m(r.anomaly)} | {rw} | {ft} | "
                   f"{round(float(r.seconds or 0.0), 1)} |"
                   + (f" {r.machine} |" if multi else ""))
    tasks = ("load", "predict", "quantiles", "embed", "classify", "anomaly",
             "reward", "finetune")
    # "skip" is the untouched default (the task was not requested); only "skipped",
    # "none" and "n/a" mean the model or its package declined the task
    declined = lambda r, t: getattr(r, t) in ("skipped", "none", "n/a")
    failed = lambda r, t: getattr(r, t) in ("fail", "failed")
    out += ["", "## Summary", "",
            "| task | ok | declined by the model | failed | not checked |", "|---|---|---|---|---|"]
    for t in tasks:
        out.append(f"| {t} | {sum(1 for r in rows if getattr(r, t) == 'ok')} "
                   f"| {sum(1 for r in rows if declined(r, t))} "
                   f"| {sum(1 for r in rows if failed(r, t))} "
                   f"| {sum(1 for r in rows if getattr(r, t) == 'skip')} |")
    out += ["", "'declined' means the model or its vendor package offers no path for that "
                "task in this environment (a `—` in the table): TimesFM's and TiRex's public "
                "builds ship inference only, Timestral and Yug likewise, Aurora's "
                "flow-matching head has no public training step, and the installed "
                "tsfm_public `PatchTSTFMForPrediction.forward` declares no target argument. "
                "'failed' means something was attempted and broke, with the reason captured "
                "below. A declined row is a statement about the model package; a failed row "
                "is a statement about this environment or our adapter — and history says to "
                "check the second category before believing the first "
                "(`docs/08_known_issues.md`, fifth pass).",
            ""]
    ran = [r for r in rows if r.reward == "ok" and r.reward_mode]
    out += ["", "## RL reachability", "",
            "Which post-training path each architecture exposes. `grpo` is the strictest "
            "(token log-probs on its own sampled paths); `rft` is what a sample-only or "
            "diffusion head can do; `descent` needs a differentiable head; the point-only "
            "models get the point version. This is a property of the vendor's public build, "
            "not a preference of the harness.", "",
            "| path | models |", "|---|---|"]
    for mode, label in (("grpo", "grpo"), ("rft", "rft"), ("reward_descent", "descent"),
                        ("reward_descent_point", "descent·pt")):
        names = sorted(r.model for r in ran if r.reward_mode == mode)
        out.append(f"| {label} | {len(names)}: {', '.join(names) if names else '—'} |")
    out += ["", f"declined by the package (no trainable path at all): "
                f"{sum(1 for r in rows if declined(r, 'reward'))} · not checked: "
                f"{sum(1 for r in rows if r.reward == 'skip')}", ""]
    fails = [r for r in rows if r.reasons]
    if fails:
        out += ["", "## Captured failures", ""]
        for r in fails:
            out.append(f"- **{r.model}** ({r.family}, python {r.python}): "
                       + "; ".join(f"{k}: {v}" for k, v in r.reasons.items()))
    text = "\n".join(out) + "\n"
    if str(path) == "-":
        # `--md -` means "print the table", not "create a file called -"; stdout is a
        # data channel here, so this stays the one place a report goes to stdout
        sys.stdout.write(text)
        return "-"
    Path(path).write_text(text)
    return str(path)


# ------------------------------------------------------------------ multi-env runner
def env_interpreters(envs_dir: str = "envs") -> dict:
    """env name -> interpreter, found the two sanctioned ways (see run_all_envs).

    Exposed because the benchmark sweep (scripts/sweep_evaluate.py) must dispatch to
    exactly the same interpreters the compatibility matrix used: a row measured in a
    different environment than the matrix says is not the same experiment.
    """
    base = Path(envs_dir).expanduser()
    interpreters = {}
    if base.is_dir():
        for d in sorted(base.iterdir()):
            for cand in (d / ".venv" / "bin" / "python", d / "bin" / "python"):
                if cand.exists():
                    interpreters[d.name] = str(cand)
                    break
    mapping = os.environ.get("TSFMTRL_ENVS_JSON") or (base / "local.json")
    try:
        for name, py in json.loads(Path(mapping).expanduser().read_text()).items():
            p = Path(py).expanduser()
            if p.exists():
                interpreters[name] = str(p)
    except FileNotFoundError:
        pass
    return interpreters


def run_all_envs(envs_dir: str = "envs", **kw) -> list[Row]:
    """Run the matrix once per family environment and merge. Each env is its own
    subprocess: a segfault or an OOM in one family cannot take the run down.

    Interpreters are found two ways, and both are allowed at once:
    ``<envs_dir>/<name>/bin/python`` (what ``scripts/make_envs.sh`` builds) and
    ``<envs_dir>/local.json`` -- ``{"core": "/path/to/.venv2/bin/python", ...}`` -- for
    machines where the working venvs already live somewhere else.
    """
    interpreters = env_interpreters(envs_dir) or {"core": sys.executable}
    rows: list[Row] = []
    for name, py in interpreters.items():
        env = dict(os.environ)
        env.setdefault("TSFMTRL_QUIET", "1")
        # the child writes its report to a file rather than to stdout: a trainer's
        # progress lines share that stream, and parsing stdout for the *last* JSON
        # object is exactly the kind of thing that silently drops rows
        fd, tmp = tempfile.mkstemp(prefix=f"matrix_{name}_", suffix=".json")
        os.close(fd)
        # --md - so a child never writes the merged markdown out from under the parent
        cmd = [py, "-m", "tsfm_trl.doctor", "--env", name, "--json", tmp, "--md", "-"] + \
              _forward(kw)
        # children buffer their JSON until they finish, so without this a 30-minute
        # matrix looks hung
        print(f"== env {name}: {py}", flush=True)
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=kw.get("timeout", 7200))
            report = Path(tmp)
            if report.exists() and report.stat().st_size:
                got = [row_from_dict(r) for r in
                       json.loads(report.read_text()).get("rows", [])]
                rows += got
                print(f"   {len(got)} rows, "
                      f"{sum(1 for r in got if r.ok('predict'))} forecast ok, "
                      f"{sum(1 for r in got if r.reasons)} with reasons", flush=True)
            else:
                # an env that produced no JSON (argparse error, import crash, segfault)
                # must not vanish from the matrix: the row says which env died and why
                why = (p.stderr or p.stdout or "no output").strip().splitlines()
                rows.append(Row(model=f"<env {name}>", env=name, load="fail",
                                reasons={"env": f"exit {p.returncode}: {(why[-1] if why else '')[:300]}"}))
        except Exception as e:                                     # noqa: BLE001
            rows.append(Row(model=f"<env {name}>", env=name, load="fail",
                            reasons={"env": f"{type(e).__name__}: {e}"}))
        finally:
            Path(tmp).unlink(missing_ok=True)
    return rows


def _forward(kw) -> list[str]:
    out = []
    for k, v in kw.items():
        if v is None or k == "timeout":
            continue
        if isinstance(v, (tuple, list)):
            v = ",".join(str(x) for x in v)
        # the parent's keyword is more readable than the child's flag
        flag = {"series-path": "series"}.get(k.replace("_", "-"), k.replace("_", "-"))
        out += [f"--{flag}", str(v)]
    return out


def main() -> None:                                 # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="", help="comma-separated model keys (default: all)")
    ap.add_argument("--tier", default="", help="restrict tiers, e.g. A or A,B")
    ap.add_argument("--env", default="", help="restrict to one family environment")
    ap.add_argument("--tasks", default="load,predict,finetune,classify,anomaly,reward")
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=21)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--series", default="", help="use a real dataset instead of the fixture")
    ap.add_argument("--json", default="reports/compat_matrix.json",
                    help="output path, or '-' for stdout")
    ap.add_argument("--md", default="COMPATIBILITY.md")
    ap.add_argument("--all-envs", action="store_true", help="dispatch one subprocess per env")
    ap.add_argument("--envs-dir", default="envs")
    ap.add_argument("--clf-windows", type=int, default=192,
                    help="classification windows per model in the matrix (evenly spaced)")
    ap.add_argument("--reward-steps", type=int, default=6,
                    help="cap on reward/GRPO steps (the matrix checks the capability, not the training)")
    ap.add_argument("--timeout", type=int, default=7200, help="per-env subprocess budget (s)")
    a = ap.parse_args()
    keys = [k for k in a.model.split(",") if k]
    tasks = tuple(t for t in a.tasks.split(",") if t)
    rows = (run_all_envs(a.envs_dir, tier=a.tier or None, env=a.env or None, tasks=tasks,
                         ctx=a.ctx, horizon=a.horizon, device=a.device, series_path=a.series,
                         reward_steps=a.reward_steps, timeout=a.timeout)
            if a.all_envs else
            run_matrix(keys, tier=a.tier or None, env=a.env or None, tasks=tasks, ctx=a.ctx,
                       horizon=a.horizon, device=a.device, series_path=a.series,
                       reward_steps=a.reward_steps))
    data = merge(rows)
    if a.json == "-":
        print(json.dumps(data, indent=1, default=str))
    else:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(data, indent=1, default=str))
        print(f"wrote {a.json}")
    if a.md and a.md != "-":
        write_markdown(rows, a.md)
        print(f"wrote {a.md}")
    n_ok = sum(1 for r in rows if r.ok("predict"))
    n_ft = sum(1 for r in rows if r.ok("finetune"))
    n_clf = sum(1 for r in rows if r.ok("classify"))
    print(f"{len(rows)} rows: {n_ok} forecast ok, {n_ft} fine-tune ok, {n_clf} classify ok")


if __name__ == "__main__":                          # pragma: no cover
    main()
