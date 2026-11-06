"""Sweep driver: plan the grid, dispatch env-isolated workers, report progress.

Everything is resumable: ``plan`` writes cells into ``reports/state.sqlite3``,
workers claim/finish them one by one, and any command can be re-run after an
interrupt.  Heavy fine-tuning is deliberately *not* run here - trainers are a
separate command (docs/05) whose checkpoints this driver then consumes.

Commands
--------
  --stage prefetch-data      download/refresh datasets (T1-T4)
  --stage derive             build derived task datasets (_dir5/_volreg/_extreme/_trade)
  --stage plan               materialise the cell grid (idempotent, --force to redo)
  --stage forecast|trading|classify|anomaly    run workers for those stages
  --stage status             print counts + write reports/PROGRESS.md
  --stage export             write reports/multi_model_results.json (legacy shape)
  --stage gc                 drop fine-tune checkpoints that are already consumed

Examples
--------
  python -m tsfm4finance.bench.sweep --stage plan --tiers A,B --groups core12,long12
  python -m tsfm4finance.bench.sweep --stage forecast --tiers A --groups core12
  python -m tsfm4finance.bench.sweep --stage forecast --env core --shard 1 --n-shards 4
  python -m tsfm4finance.bench.sweep --stage status
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from itertools import product
from pathlib import Path

from .config import load_grid
from .state import CellStore

REPO = Path(__file__).resolve().parents[2]
DB = REPO / "reports" / "state.sqlite3"
FT_DIR = REPO / "reports" / "ft_ckpts"
RUN_STAGES = ("forecast", "trading", "classify", "anomaly")


# ------------------------------------------------------------------- data prep
def stage_prefetch_data(a) -> None:
    from tsfm4finance.bench.download import prefetch
    res = prefetch(tiers=tuple(t.strip() for t in a.tiers.split(",")) if a.tiers
                   else ("t1", "t2", "t3", "t4"), force=a.force)
    print(json.dumps(res, indent=1, default=str))


def stage_derive(a) -> None:
    """Derived task datasets for every base series available on disk."""
    from tsfm4finance.bench.download.common import dataset_names
    from tsfm4finance.bench.tasks import build_for_series
    grid = load_grid(a.config)
    names = set()
    for g in (a.groups.split(",") if a.groups else list(grid.series_groups())):
        names |= set(grid.series(groups=(g,)))
    from tsfm4finance.bench.download.common import load_dataset
    valid = {d.value for d in __import__("tsfm4finance.core.schemas",
                                         fromlist=["Domain"]).Domain}
    created, skipped = [], []
    for n in sorted(names):
        if n.endswith(("_dir5", "_volreg", "_extreme", "_trade")):
            continue
        try:
            _tr, _va, _te, meta = load_dataset(n)
            dom = str(meta.get("domain") or "general").lower()
            got = build_for_series(n, domain=dom if dom in valid else "general")
            created += got
        except Exception as e:                                     # noqa: BLE001
            skipped.append((n, f"{type(e).__name__}: {e}"[:160]))
    print(json.dumps({"derived": len(created), "failed": len(skipped),
                      "failures": skipped[:12]}, indent=1))


# ----------------------------------------------------------------------- plan
def ft_checkpoints() -> dict[tuple[str, str], str]:
    """``(model, budget) -> checkpoint path`` for what the trainers produced."""
    out: dict[tuple[str, str], str] = {}
    if not FT_DIR.exists():
        return out
    for p in list(FT_DIR.rglob("*.pt")) + list(FT_DIR.rglob("*.safetensors")):
        stem = p.name.rsplit(".", 1)[0]
        parts = stem.split("__")
        if len(parts) >= 2:
            out[(parts[0], parts[-1])] = str(p)
    return out


def stage_plan(a) -> None:
    grid = load_grid(a.config)
    store = CellStore(a.db)
    sw = grid.section("sweep") or {}
    horizons = [int(h) for h in (a.horizons.split(",") if a.horizons else sw.get("horizons", [21]))]
    contexts = [int(c) for c in (a.contexts.split(",") if a.contexts else sw.get("contexts", [512]))]
    tiers = tuple(t.strip() for t in a.tiers.split(",") if t.strip()) or None
    groups = tuple(g.strip() for g in a.groups.split(",") if g.strip()) or None
    seeds = [int(x) for x in (a.seeds.split(",") if a.seeds else sw.get("seeds", [0]))]
    adaptations = (tuple(a.adaptations.split(",")) if a.adaptations
                   else tuple(sw.get("adaptations", ["zero"])))
    spans = a.spans or sw.get("spans", "auto")
    models = grid.models(tiers=tiers,
                         only=tuple(m.strip() for m in a.models.split(",")) if a.models else None)
    have = dataset_dir_names()
    series = ([x.strip() for x in a.series.split(",") if x.strip()] if a.series
              else [x for x in grid.series(groups=groups) if x in have or not have])
    ck = ft_checkpoints()
    ch = grid.digest()
    trade_h = [int(h) for h in (sw.get("trading", {}) or {}).get("rebalance_horizons") or [5]]
    routes_clf = list(sw.get("classify_routes") or [])
    routes_an = list(sw.get("anomaly_routes") or [])

    specs: list[dict] = []

    def add(stage, series_name, model, adaptation, horizon=None, span=None, seed=0):
        specs.append({"stage": stage, "series": series_name, "model": model,
                      "adaptation": adaptation, "horizon": horizon, "span": span,
                      "seed": seed, "config_hash": ch})

    for mkey, spec in sorted(models.items()):
        can_fc = "forecast" in spec.caps
        # 1) forecasting grid: every series x context x horizon the model can serve
        if can_fc:
            for s, ctx, seed, h in product(series, contexts, seeds, horizons):
                if h > spec.max_horizon and not spec.can("sample_paths"):
                    continue
                add("forecast", s, mkey, "zero", h, spans, seed)
        # 2) fine-tuned variants, but only for checkpoints the trainers actually made
        if can_fc:
            for budget in adaptations:
                if budget == "zero" or not spec.can("finetune"):
                    continue
                for s in series:
                    if (mkey, budget, s) not in ck and (mkey, budget) not in ck:
                        continue
                    for h in horizons:
                        add("forecast", s, mkey, budget, h, spans, 0)
        # 3) trading: the same forecasts priced through a cost model
        if can_fc:
            for s in series:
                if f"{s}_trade" not in have:
                    continue
                for h in trade_h:
                    add("trading", f"{s}_trade", mkey, "zero", h, spans, 0)
        # 4) classification probes (the statistical route is planned once, as `naive`)
        for s in series:
            for task_ds in (f"{s}_dir5", f"{s}_volreg"):
                if task_ds not in have:
                    continue
                for route in routes_clf:
                    cap = {"native_head": "clf_head", "embedding_probe": "embed"}.get(route)
                    if cap and not spec.can(cap):
                        continue
                    if route == "stat" and mkey != "naive":
                        continue
                    add("classify", task_ds, mkey, "zero", 1, f"probe:{route}", 0)
            # 5) anomaly detection on real extreme-move labels
            if f"{s}_extreme" not in have:
                continue
            for route in routes_an:
                cap = {"band_violation": "quantiles",
                       "native_reconstruction": "anomaly_head"}.get(route)
                if cap and not spec.can(cap):
                    continue
                if route == "zscore" and mkey != "naive":
                    continue
                add("anomaly", f"{s}_extreme", mkey, "zero", 1, route, 0)

    todo, done, other = store.plan(specs, force=a.force)
    store.put_meta("grid_hash", ch)
    print(json.dumps({"planned_total": len(specs), "new_todo": len(todo),
                      "already_done": done, "other_terminal": other,
                      "models": len(models), "series": len(series),
                      "datasets_on_disk": len(have),
                      "contexts": contexts, "horizons": horizons}, indent=1))


def dataset_dir_names() -> set[str]:
    from tsfm4finance.bench.download.common import dataset_names
    try:
        return set(dataset_names())
    except Exception:                                              # noqa: BLE001
        root = REPO / "data" / "open"
        return {p.name for p in root.iterdir() if p.is_dir()} if root.exists() else set()


# -------------------------------------------------------------------- dispatch
def stage_run(a) -> None:
    """Spawn one worker process per (env, stage, shard); env isolation is the point."""
    grid = load_grid(a.config)
    store = CellStore(a.db)
    store.reset_running()
    tiers = tuple(t.strip() for t in a.tiers.split(",") if t.strip()) or None
    envs = [a.env] if a.env else sorted({m.env for m in grid.models(tiers=tiers).values()})
    stages = list(a.run_stages) or ["forecast"]
    cmds = []
    for stage, env in product(stages, envs):
        for shard in range(a.n_shards):
            cmds.append([sys.executable, "-m", "tsfm4finance.bench.cell_worker",
                         "--stage", stage, "--env", env, "--shard", str(shard),
                         "--n-shards", str(a.n_shards), "--db", str(a.db),
                         "--device", a.device, "--amp", a.amp, "--batch", str(a.batch),
                         "--ctx-cap", str(a.ctx_cap)])
    if a.dry_run:
        print(json.dumps({"would_run": cmds[:12], "n": len(cmds)}))
        return
    running: list[tuple[subprocess.Popen, list[str]]] = []
    t0 = time.time()
    for cmd in cmds:
        while len([p for p, _ in running if p.poll() is None]) >= a.workers:
            time.sleep(2.0)
        print("+ " + " ".join(cmd[3:]), flush=True)
        running.append((subprocess.Popen(cmd, cwd=str(REPO)), cmd))
    bad = 0
    for p, cmd in running:
        rc = p.wait()
        if rc != 0:
            bad += 1
            print(f"worker exited rc={rc}: {' '.join(cmd[3:])}", flush=True)
    print(json.dumps({"workers": len(cmds), "failed": bad,
                      "seconds": round(time.time() - t0, 1),
                      "counts": store.counts()}, indent=1))


# ----------------------------------------------------------------- status/report
STATUS_ORDER = ("done", "pending", "running", "skipped", "failed")


def stage_status(a) -> None:
    grid = load_grid(a.config)
    store = CellStore(a.db)
    rows = store.rows()
    by_model: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for r in rows:
        by_model[r["model"]][r["stage"]][r["status"]] += 1
    lines = ["# Benchmark progress", "",
             f"grid: `{grid.digest()}`  cells: `{len(rows)}`  "
             f"store: `{str(a.db.relative_to(REPO))}`", ""]
    counts = store.counts()
    lines += ["| stage | " + " | ".join(STATUS_ORDER) + " |",
              "|---|" + "---:|" * len(STATUS_ORDER)]
    for stage in sorted(counts):
        c = counts[stage]
        lines.append(f"| {stage} | " + " | ".join(str(c.get(k, 0)) for k in STATUS_ORDER) + " |")
    lines += ["", "## Per model", "",
              "| model | tier | env | stage | " + " | ".join(STATUS_ORDER) + " | headline |",
              "|---|:--|:--|:--|" + "---:|" * len(STATUS_ORDER) + "|---|"]
    specs = grid.models()
    for m in sorted(by_model):
        spec = specs.get(m)
        for stage in sorted(by_model[m]):
            c = by_model[m][stage]
            head = _headline(store, m, stage)
            lines.append(f"| {m} | {getattr(spec, 'tier', '?')} | {getattr(spec, 'env', '?')} "
                         f"| {stage} | " + " | ".join(str(c.get(k, 0)) for k in STATUS_ORDER)
                         + f" | {head} |")
    # explicit pending/failed inventory so nothing is silently missing
    lines += ["", "## Not completed", ""]
    for r in rows:
        if r["status"] == "failed":
            lines.append(f"- FAILED `{r['stage']}` `{r['series']}` `{r['model']}` "
                         f"h={r['horizon']}: {str(r['error'])[:120]}")
        elif r["status"] == "skipped":
            lines.append(f"- SKIPPED `{r['stage']}` `{r['series']}` `{r['model']}`: "
                         f"{str(r['reason'])[:120]}")
    out = REPO / "reports" / "PROGRESS.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:60]))
    print(f"\n[written] {out}")


def _headline(store: CellStore, model: str, stage: str) -> str:
    if stage != "forecast":
        return ""
    vals = []
    for r in store.rows(stage="forecast", status="done"):
        if r["model"] != model or int(r["horizon"] or 0) != 21:
            continue
        p = store.payload(r["cell_id"]) or {}
        v = (p.get("aggregate") or {}).get("mase")
        if v is not None:
            vals.append(float(v))
    return f"mean MASE@21 = {sum(vals) / len(vals):.3f} (n={len(vals)})" if vals else ""


def stage_export(a) -> None:
    store = CellStore(a.db)
    data = store.export()
    out = REPO / "reports" / "multi_model_results.json"
    out.write_text(json.dumps(data, indent=1, default=str))
    print(json.dumps({k: len(v) for k, v in data.items()}, indent=1))
    print(f"[written] {out}")


def stage_gc(a) -> None:
    """Fine-tune checkpoints are consumables: delete what has been evaluated."""
    store = CellStore(a.db)
    done = {(r["model"], r["adaptation"]) for r in store.rows(status="done")
            if r["adaptation"] != "zero"}
    n = 0
    if FT_DIR.exists():
        for p in FT_DIR.iterdir():
            stem = p.name.rsplit(".", 1)[0]
            parts = stem.split("__")
            if len(parts) >= 2 and (parts[0], parts[-1]) in done and p.is_file():
                p.unlink()
                n += 1
    print(json.dumps({"removed_checkpoints": n}))


STAGES = {"prefetch-data": stage_prefetch_data, "derive": stage_derive, "plan": stage_plan,
          "status": stage_status, "export": stage_export, "gc": stage_gc, **{
              s: stage_run for s in RUN_STAGES}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=False, choices=sorted(STAGES))
    ap.add_argument("--config", default=None)
    ap.add_argument("--db", default=str(DB))
    ap.add_argument("--tiers", default="", help="e.g. A or A,B (default: all)")
    ap.add_argument("--groups", default="", help="series_groups keys, e.g. core12,long12")
    ap.add_argument("--models", default="", help="restrict to these model keys")
    ap.add_argument("--series", default="", help="restrict to these datasets")
    ap.add_argument("--horizons", default="", help="default: sweep.horizons from config")
    ap.add_argument("--contexts", default="")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--spans", default="")
    ap.add_argument("--adaptations", default="", help="default: sweep.adaptations")
    ap.add_argument("--env", default="", help="run only this env's workers")
    ap.add_argument("--stage-run", dest="run_stages", action="append", default=[],
                    help="with --stage forecast|... : additionally run these stages")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", default="fp32")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ctx-cap", type=int, default=512)
    ap.add_argument("--force", action="store_true", help="re-run terminal cells")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tiers-list", action="store_true", help="print the model registry and exit")
    a = ap.parse_args()

    if a.tiers_list:
        a.stage = a.stage or "status"
        g = load_grid(a.config)
        for k, m in sorted(g.models().items()):
            print(f"{k:<22} tier={m.tier} env={m.env:<7} params={m.params_m:>7} "
                  f"caps={','.join(m.caps):<48} {m.weights}")
        return
    if not a.stage:
        ap.error("--stage is required (or use --tiers-list)")
    if a.stage in RUN_STAGES:
        a.run_stages = list(dict.fromkeys(list(a.run_stages) + [a.stage]))
    STAGES[a.stage](a)


if __name__ == "__main__":
    main()
