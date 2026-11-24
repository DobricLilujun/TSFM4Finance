"""``tsfm-trl`` command line.

    tsfm-trl list  [--tier A] [--task forecast] [--family chronos] [--short]
    tsfm-trl info  <model>
    tsfm-trl doctor [--model m,...] [--tier A,B] [--tasks load,predict,...] [--all-envs]
    tsfm-trl forecast --model chronos-2 --series <dir|csv> [--horizon 21] [--ctx 256]
    tsfm-trl fit      --model timesfm-2.5 --series <dir> [--budget fs16|fs64|full] [--peft ...]
                      [--steps N] [--smoke] [--out reports/ft]
    tsfm-trl classify --model moment-1-large --series <dir> [--route head|probe|native]
                      [--label dir|extreme|volreg|<csv>]
    tsfm-trl reward   --model chronos-t5-small --series <dir> [--mode auto] [--steps 60]
    tsfm-trl evaluate --model timesfm-2.5 --series <dir> [--adaptation zero|<ckpt>]
    tsfm-trl ckpt     [--ls reports/ft]

Model names are keys from ``tsfm_trl/models.yaml``; ``--weights`` overrides.
"""
from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path



def _series_arg(a):
    from .data.series import SeriesDataset
    path = getattr(a, "series", "") or ""
    if not path.strip():
        # no --series means "show me that this works", with the tiny synthetic
        # fixture the matrix uses; say so on stderr so nobody mistakes the number
        # for a result on their own data
        import sys
        from .data.fixtures import tiny_finance
        print("tsfm-trl: no --series given; using the built-in fixture (tiny_fin)",
              file=sys.stderr)
        return tiny_finance()
    return SeriesDataset.load(path)


def _load_model(a):
    from .model import ModelHub
    return ModelHub.load(a.model, checkpoint=getattr(a, "weights", None), device=a.device,
                         batch=a.batch, seed=a.seed)


def cmd_list(a):
    from .model import ModelHub
    rows = ModelHub.list(tier=a.tier, family=a.family, task=a.task)
    for s in rows:
        if a.short:
            print(f"{s.key:26s} {s.family:14s} tier {s.tier}  env {s.env}")
        else:
            print(f"{s.key:26s} {s.family:14s} tier {s.tier}  env {s.env:10s} "
                  f"loss {str(s.loss or '-'):8s} trainable {'y' if s.trainable else 'n':>9s}  "
                  f"{s.license:22s} {s.weights}")
    print(f"\n{len(rows)} models")
    return 0


def cmd_info(a):
    from .model import ModelHub
    s = ModelHub.info(a.model)
    print(json.dumps({"key": s.key, "family": s.family, "tier": s.tier, "env": s.env,
                      "weights": s.weights, "license": s.license, "caps": sorted(s.caps),
                      "loss_kind": s.loss, "trainable": s.trainable, "max_context": s.max_context,
                      "max_horizon": s.max_horizon, "notes": s.notes,
                      "capabilities": sorted(__import__("tsfm_trl.model.base",
                                                        fromlist=["CAPS"]).CAPS)}, indent=1))
    return 0


def cmd_doctor(a):
    from .doctor import (dedupe, merge, row_from_dict, run_all_envs, run_matrix,
                         write_markdown)
    if a.merge:
        rows = []
        for f in a.merge:
            rows += [row_from_dict(r) for r in
                     json.loads(Path(f).read_text()).get("rows", [])]
        raw = len(rows)
        # a row whose env no longer matches the registry is a stale artifact of an
        # earlier run (the model was moved to a different environment); reporting it
        # next to the current one would double-count the model
        from tsfm_trl import ModelHub
        cur = {}
        for r in rows:
            try:
                cur[r.model] = ModelHub.lookup(r.model).env
            except Exception:                                      # noqa: BLE001
                cur[r.model] = r.env
        moved = [r.model for r in rows if cur.get(r.model) and r.env != cur[r.model]]
        if moved and not a.keep_both:
            rows = [r for r in rows if r.env == cur.get(r.model)]
            print(f"dropped {len(moved)} row(s) whose env moved in the registry: "
                  f"{sorted(set(moved))}")
        rows = dedupe(rows) if not a.keep_both else rows
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(merge(rows), indent=1, default=str))
        md = a.md or "COMPATIBILITY.md"
        write_markdown(rows, md)
        print(f"merged {len(a.merge)} reports -> {raw} rows, {len(rows)} distinct "
              f"(model, env): {a.json}, {md}")
        return 0
    keys = [k for k in a.models.split(",") if k]
    tasks = tuple(t for t in a.tasks.split(",") if t)
    kw = dict(tier=a.tier or None, env=a.env or None, tasks=tasks, ctx=a.ctx,
              horizon=a.horizon, device=a.device, series_path=a.series or "",
              reward_steps=a.reward_steps, clf_windows=a.clf_windows, timeout=a.timeout)
    # run_all_envs selects the rows by env itself; passing --model keys would make every
    # env re-run the same rows in the wrong interpreter
    rows = run_all_envs(a.envs_dir, **kw) if a.all_envs else run_matrix(keys, **kw)
    data = merge(rows)
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(data, indent=1, default=str))
    if a.md:
        write_markdown(rows, a.md)
    print(f"wrote {a.json}" + (f" and {a.md}" if a.md else ""))
    print(f"{len(rows)} rows: {sum(1 for r in rows if r.ok('predict'))} forecast ok, "
          f"{sum(1 for r in rows if r.ok('finetune'))} fine-tune ok, "
          f"{sum(1 for r in rows if r.ok('classify'))} classify ok")
    for r in rows:
        if r.reasons:
            print(f"  {r.model}: " + "; ".join(f"{k}: {v[:120]}" for k, v in r.reasons.items()))
    return 0


def cmd_forecast(a):
    from .eval.evaluate import forecast_eval
    model = _load_model(a)
    ds = _series_arg(a)
    out = forecast_eval(model, ds, horizons=(a.horizon,), ctx=a.ctx,
                        max_windows=a.max_windows, batch=a.batch, seed=a.seed)
    if not a.md:
        print(json.dumps(out, indent=1, default=float))
        return 0
    for r in out:
        print(f"### {r['model']} · {r['series']} · {r['span']} · h={r['horizon']}\n")
        print(f"probabilistic: {'yes' if 'crps' in r else 'no (point-only model)'} · ctx {r['ctx']}"
              f"{(' (' + r['ctx_note'] + ')') if r.get('ctx_note') else ''} · "
              f"{r['n_windows']} windows · {r['seconds']}s\n")
        print("| RMSE | MAE | MASE | sMAPE | ND | dir-acc | rank_corr | skill vs "
              "seasonal | CRPS | WQL | pinball |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        def f(v):                    # nan/None render as an em dash rather than 'nan'
            return "—" if v is None or not isinstance(v, (int, float)) or v != v else f"{v:.4g}"
        print(f"| {f(r['rmse'])} | {f(r['mae'])} | {f(r['mase'])} | {f(r['smape'])} | "
              f"{f(r['nd'])} | {f(r['dir_acc'])} | {f(r['rank_corr'])} | "
              f"{f(r['relative_skill'])} | {f(r.get('crps'))} | {f(r.get('wql'))} | "
              f"{f(r.get('pinball'))} |")
    return 0


def cmd_fit(a):
    from .data.series import train_windows
    from .training.config import TrainingConfig
    from .training.forecast import ForecastTrainer
    model = _load_model(a)
    ds = _series_arg(a)
    w, info = train_windows(ds, a.horizon, a.ctx, max_windows=a.max_windows)
    cfg = TrainingConfig(ctx=a.ctx, horizon=a.horizon, budget=a.budget, steps=a.steps, bs=a.bs,
                         lr=a.lr, peft=a.peft, device=a.device, seed=a.seed, smoke=a.smoke,
                         out_dir=a.out, tag=a.model,
                         lora_targets=tuple(t for t in a.lora_targets.split(",") if t))
    res = ForecastTrainer(model, cfg).fit({"windows": w, "info": info})
    print(json.dumps({"status": res.status, "steps": res.steps, "seconds": res.seconds,
                      "first_loss": res.first_loss, "final_loss": res.final_loss,
                      "trainable_parameters": res.trainable_parameters,
                      "checkpoint": res.checkpoint,
                      # which (bs, amp, peft) profile actually ran, if the first one OOMed
                      "fallbacks": res.extra.get("fallbacks", []),
                      "reason": res.reason,
                      "trained_parameters": res.trained_parameters[:8],
                      "mode": res.extra.get("mode", "")}, indent=1, default=float))
    return 0 if res.status in ("ok", "smoke") else 1


def cmd_classify(a):
    from .data.classify import direction_dataset, extreme_dataset, label_dataset, volreg_dataset
    from .training.classify import ClassificationTrainer
    from .training.config import TrainingConfig
    ds = _series_arg(a)
    if a.label in ("dir", "dir5"):
        cds = direction_dataset(ds, horizon=a.horizon, ctx=a.ctx,
                                levels=5 if a.label == "dir5" else 3)
    elif a.label == "extreme":
        cds = extreme_dataset(ds, horizon=a.horizon, ctx=a.ctx)
    elif a.label == "volreg":
        cds = volreg_dataset(ds, horizon=a.horizon, ctx=a.ctx)
    else:
        import pandas as pd
        lab = pd.read_csv(a.label).iloc[:, 1].to_numpy()
        cds = label_dataset(ds, lab, ctx=a.ctx)
    model = _load_model(a)
    cfg = TrainingConfig(ctx=a.ctx, horizon=a.horizon, budget=a.budget, steps=a.steps, bs=a.bs,
                         lr=a.lr, peft=a.peft, device=a.device, seed=a.seed, smoke=a.smoke,
                         out_dir=a.out, tag=a.model)
    routes = [r for r in (a.routes or "head").split(",") if r]
    if a.route:
        routes = [a.route]
    # one row per route: a TSFM is only interesting if it beats the frozen probe and the
    # statistical baseline, so comparing one route at a time invites cherry-picking
    worst = 0
    for route in routes:
        res = ClassificationTrainer(model, cfg, route=route).fit(cds)
        print(json.dumps({"status": res.status, "route": route, "label": cds.name,
                          "n_classes": cds.n_classes, "steps": res.steps,
                          "seconds": res.seconds, "metrics": res.extra.get("metrics"),
                          "checkpoint": res.checkpoint, "reason": res.reason},
                         indent=1, default=float))
        if res.status not in ("ok", "smoke"):
            worst = 1
    return worst


def cmd_reward(a):
    from .data.series import train_windows
    from .training.config import TrainingConfig
    from .training.reward import RewardSpec, RewardTrainer
    model = _load_model(a)
    ds = _series_arg(a)
    w, info = train_windows(ds, a.horizon, a.ctx, max_windows=a.max_windows)
    cfg = TrainingConfig(ctx=a.ctx, horizon=a.horizon, budget=a.budget, steps=a.steps, bs=a.bs,
                         lr=a.lr, peft=a.peft, device=a.device, seed=a.seed, smoke=a.smoke,
                         out_dir=a.out, tag=a.model)
    # kl belongs to the trainer (it anchors the policy to the reference weights);
    # the spec only carries the reward weights
    tr = RewardTrainer(model, cfg, spec=RewardSpec(w_pinball=a.pinball_weight),
                       kl=a.kl, n_candidates=a.candidates, mode=a.mode)
    res = tr.fit({"windows": w, "info": info})
    rf, rl = (res.extra.get("reward_first_last") or [None, None])
    print(json.dumps({"status": res.status, "mode": res.extra.get("mode"), "steps": res.steps,
                      "reward_first": rf, "reward_last": rl,
                      "reward_early_mean": res.extra.get("reward_early_mean"),
                      "reward_late_mean": res.extra.get("reward_late_mean"),
                      "improved": res.extra.get("improved"),
                      "fallbacks": res.extra.get("fallbacks"),
                      "mean_loss": res.final_loss, "reason": res.reason,
                      "trainable_parameters": res.trainable_parameters,
                      "checkpoint": res.checkpoint}, indent=1, default=float))
    return 0 if res.status in ("ok", "smoke") else 1


def cmd_evaluate(a):
    from .eval.evaluate import forecast_eval
    model = _load_model(a)
    if a.adaptation != "zero":
        from .training.checkpoints import load_finetuned
        model = load_finetuned(model, a.adaptation)
    hs = tuple(int(x) for x in a.horizons.split(",") if x.strip()) or (a.horizon,)
    out = forecast_eval(model, _series_arg(a), horizons=hs, ctx=a.ctx,
                        spans=tuple(x.strip() for x in a.spans.split(",") if x.strip()),
                        max_windows=a.max_windows, seed=a.seed)
    if getattr(a, "out", ""):
        # the sweep driver collects many models: the child writes a file, and stdout
        # stays a log channel that a stray tqdm line cannot corrupt
        Path(a.out).write_text(json.dumps(out, indent=1, default=float))
        print(f"wrote {a.out} ({len(out)} rows)", file=sys.stderr)
    else:
        print(json.dumps(out, indent=1, default=float))
    return 0


def cmd_ckpt(a):
    from .training.checkpoints import describe
    root = Path(a.ls)
    if not root.exists():
        print(f"{root} does not exist yet")
        return 1
    for p in sorted(root.rglob("model.pt")):
        md = describe(p)
        print(f"{p}\n    base={md['base']} steps={md['steps']} budget={md['budget']} "
              f"trainable={md['trainable']} split_end={md['split_end']}")
        if a.json:
            print(json.dumps(md, indent=1, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="tsfm-trl", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, model=True, series=True, train=True, device=True):
        if model:
            p.add_argument("--model", required=True)
            p.add_argument("--weights", default=None)
        if series:
            p.add_argument("--series", default="",
                           help="dataset dir, .csv or .npz (omit for the built-in "
                                "tiny fixture — a capability demo, not your data)")
        p.add_argument("--horizon", type=int, default=21)
        p.add_argument("--ctx", type=int, default=256)
        p.add_argument("--max-windows", type=int, default=128,
                       help="cap on windows sampled from the train split")
        if train:
            p.add_argument("--budget", default="fs16", choices=["fs16", "fs64", "full"])
            p.add_argument("--steps", type=int, default=0)
            p.add_argument("--bs", type=int, default=0)
            p.add_argument("--lr", type=float, default=0.0)
            p.add_argument("--peft", default="full", choices=["full", "heads_only", "lora"])
            p.add_argument("--lora-targets", default="",
                           help="comma list of Linear leaf names for peft=lora "
                                "(empty = the family's own hints, then a broad default)")
            p.add_argument("--smoke", action="store_true")
            p.add_argument("--out", default="reports/ft")
        if device:
            p.add_argument("--device", default="auto")
        p.add_argument("--batch", type=int, default=4)
        p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("list", help="list the model zoo")
    p.add_argument("--tier", default=None); p.add_argument("--family", default=None)
    p.add_argument("--task", default=None); p.add_argument("--short", action="store_true")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("info", help="one model's registry row")
    p.add_argument("model"); p.set_defaults(fn=cmd_info)

    p = sub.add_parser("doctor", help="run the compatibility matrix")
    p.add_argument("--model", dest="models", default="")
    p.add_argument("--tier", default=""); p.add_argument("--env", default="")
    p.add_argument("--tasks", default="load,predict,finetune,classify,anomaly,reward")
    p.add_argument("--ctx", type=int, default=256); p.add_argument("--horizon", type=int, default=21)
    p.add_argument("--device", default="auto"); p.add_argument("--series", default="")
    p.add_argument("--json", default="reports/compat_matrix.json")
    p.add_argument("--clf-windows", type=int, default=192,
                   help="classification windows per model in the matrix")
    p.add_argument("--keep-both", action="store_true",
                   help="with --merge: keep one row per machine instead of one per (model, env)")
    p.add_argument("--merge", nargs="*", default=None, metavar="REPORT.json",
                   help="merge existing report JSONs into one matrix instead of running models")
    p.add_argument("--reward-steps", type=int, default=6,
                   help="cap on reward/GRPO steps in the matrix (capability check, not a training run)")
    p.add_argument("--md", default="")
    p.add_argument("--all-envs", action="store_true",
                   help="dispatch one subprocess per family env (see envs/local.json)")
    p.add_argument("--envs-dir", default="envs")
    p.add_argument("--timeout", type=int, default=7200, help="per-env subprocess budget (s)")
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser("forecast", help="zero-shot forecast + metrics")
    common(p, train=False)
    p.add_argument("--md", action="store_true", help="markdown instead of JSON")
    p.set_defaults(fn=cmd_forecast)

    p = sub.add_parser("fit", help="fine-tune for forecasting")
    common(p); p.set_defaults(fn=cmd_fit)

    p = sub.add_parser("classify", help="fine-tune / probe for classification")
    common(p)
    p.add_argument("--routes", default="head",
                   help="comma list of routes: head (new head), probe (frozen features), "
                        "native (the family's own head). All three is the honest default "
                        "when comparing models.")
    p.add_argument("--route", default=None, choices=["head", "probe", "native"],
                   help="deprecated alias for --routes with one entry")
    p.add_argument("--label", default="dir",
                   help="dir | dir5 | extreme | volreg | path/to/labels.csv")
    p.set_defaults(fn=cmd_classify)

    p = sub.add_parser("reward", help="reward post-training (RFT/GRPO/reward-descent)")
    common(p)
    p.add_argument("--mode", default="auto",
                   choices=["auto", "rft", "grpo", "reward_descent", "reward_descent_point", "none"])
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--kl", type=float, default=0.02)
    p.add_argument("--pinball-weight", type=float, default=1.0)
    p.set_defaults(fn=cmd_reward)

    p = sub.add_parser("evaluate", help="metrics for a base or fine-tuned checkpoint")
    common(p, train=False)
    p.add_argument("--adaptation", default="zero", help="'zero' or a checkpoint path")
    p.add_argument("--out", default="", help="write the JSON rows to this file")
    p.add_argument("--spans", default="test",
                   help="comma list of spans to score: train,val,test,all — 'all' slides "
                        "windows over the whole series, which is legitimate for zero-shot "
                        "rows (the model saw no data) but NOT for adapted checkpoints")
    p.add_argument("--horizons", default="",
                   help="comma list, e.g. 21,63,126 — one evaluation pass per horizon "
                        "(defaults to the single --horizon)")
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("ckpt", help="list fine-tuned checkpoints with provenance")
    p.add_argument("--ls", default="reports/ft"); p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_ckpt)
    return ap


def main(argv=None) -> int:
    """Console entry point. ``argv`` defaults to ``sys.argv[1:]``; from a REPL call
    ``main(["list", "--short"])``."""
    a = build_parser().parse_args(argv)
    return int(a.fn(a) or 0)


if __name__ == "__main__":                          # pragma: no cover
    raise SystemExit(main())
