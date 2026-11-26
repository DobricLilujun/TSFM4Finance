#!/usr/bin/env python3
"""Benchmark sweep: the same forecast protocol over every registered model.

This is the *effectiveness* half of the library (docs/09_benchmark.md) — the
compatibility matrix answers "does it run", this answers "how well does it forecast",
on the same series, the same test span, the same horizon grid, the same window count.

Dispatch is per family environment through the same interpreter resolver the matrix
uses, and every model runs in its own subprocess: an OOM or segfault in one family is
one failed row, not a lost sweep. Rows are appended to JSONL as they land, so an
interrupted sweep keeps what it measured, and `--report` can be re-run over partial
data.

    scripts/sweep_evaluate.py --series ~/data/raw/btc.csv --series ~/data/raw/gbpusd.csv \\
        --horizons 21,63,126 --max-windows 16 --out reports/effectiveness.jsonl
    scripts/sweep_evaluate.py --report reports/effectiveness.jsonl --md EFFECTIVENESS.md
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tsfm_trl.doctor import env_interpreters                      # noqa: E402
from tsfm_trl.model.registry import default_registry              # noqa: E402

BENCH_COLS = ("rmse", "mae", "mase", "dir_acc", "rank_corr", "relative_skill")
PROB_COLS = ("crps", "wql", "pinball")


def _models(reg, spec: str, tier: str) -> list[str]:
    keys = list(reg.keys())
    if spec and spec != "all":
        want = [m.strip() for m in spec.split(",") if m.strip()]
        unknown = [m for m in want if m not in set(keys)]
        if unknown:
            raise SystemExit(f"unknown model(s): {unknown}")
        return want
    if tier:
        return [k for k in keys if reg.get(k).tier == tier]
    return keys


def sweep(args) -> int:
    reg = default_registry()
    interp = env_interpreters(args.envs)
    keys = _models(reg, args.models, args.tier)
    done = set()
    out = Path(args.out)
    # A run directory does not exist until this line, and the sweep only opens it at the
    # end: discovering that with FileNotFoundError after a lane has scored every series
    # wastes machine minutes that a shared cluster will not give back.
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.fresh:
        for line in out.read_text().splitlines():
            try:
                done.add(json.loads(line).get("model", ""))
            except json.JSONDecodeError:
                continue
    fh = out.open("a" if not args.fresh else "w")
    series = [os.path.expanduser(s) for s in args.series]

    def _sha() -> str:
        root = Path(__file__).resolve().parent.parent
        try:
            return subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:                                      # noqa: BLE001
            pass
        # most remote workers get an rsync'd tree with no .git: the sync writes REVISION
        rev = root / "REVISION"
        if rev.exists():
            return rev.read_text().strip()[:12]
        return "?"

    # A published table has to say which machine and which code produced each row;
    # without this line the same model measured in two venvs is one indistinguishable
    # number. Written once, as the first record of the file.
    if fh.tell() == 0:
        fh.write(json.dumps({"meta": {
            "started": time.strftime("%Y-%m-%d %H:%M:%S"), "machine": socket.gethostname(),
            "code_sha": _sha(), "python": sys.version.split()[0],
            "series": [os.path.basename(x) for x in series],
            "horizons": [int(h) for h in args.horizons.split(",")],
            "spans": args.spans, "max_windows": int(args.max_windows),
            "ctx": int(args.ctx), "batch": int(args.batch), "seed": int(args.seed),
            "device": args.device}}, default=float) + "\n")
        fh.flush()
    # Does this env's torch even see a GPU? Recording it turns "device auto" into a
    # claim: a family env built from a lock that resolved a CPU wheel silently runs the
    # whole benchmark on CPU, and a CUDA failure then reads like a model problem.
    probe = (
        "import json,torch;"
        "print(json.dumps({'torch': torch.__version__,"
        " 'cuda': bool(torch.cuda.is_available()),"
        " 'gpu': (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)}))"
    )

    def torch_of(py: str) -> dict:
        if py in _torch_cache:
            return _torch_cache[py]
        try:
            r = subprocess.run([py, "-c", probe], capture_output=True, text=True, timeout=120)
            out = _torch_cache.setdefault(py, json.loads(r.stdout.strip().splitlines()[-1])
                                          if r.returncode == 0 else
                                          {"torch": "?", "cuda": False,
                                           "gpu": (r.stderr or "").strip().splitlines()[-1:][0][:60]})
        except Exception as e:                                 # noqa: BLE001
            out = _torch_cache.setdefault(py, {"torch": "?", "cuda": False, "gpu": str(e)[:60]})
        return out

    _torch_cache: dict = {}
    n_ok = n_bad = skipped = 0
    for k in keys:
        if k in done:
            continue
        spec = reg.get(k)
        # No silent fallback to `core`: a family run in an interpreter it was never
        # installed into reports a foreign environment's failure (or its numbers) as if
        # they belonged to the family. Missing env => a recorded skip.
        py = interp.get(spec.env)
        if not py and args.allow_core_fallback:
            py = interp.get("core") or sys.executable
        if not py:
            why = (f"no interpreter for env '{spec.env}' on this machine: row skipped, "
                   f"not measured (the matrix ran it in {spec.env})")
            print(f"== {k}: {why}", flush=True)
            fh.write(json.dumps({"model": k, "env": spec.env, "tier": spec.tier,
                                 "family": spec.family, "rows": [], "error": why,
                                 "interpreter": ""}) + "\n")
            fh.flush()
            skipped += 1
            continue
        py_abs = _abspath(py)
        if py_abs != py:
            # provenance has to name the interpreter that ran, not one that happens to
            # share the name here
            py = py_abs
        tt = torch_of(py)
        if args.device.startswith("cuda") and not tt.get("cuda"):
            print(f"  NOTE {k}: --device cuda requested, but env '{spec.env}' "
                  f"(torch {tt.get('torch')}) reports no CUDA — this run will fail or "
                  f"fall back; not a model verdict", flush=True)
        print(f"== {k} (env {spec.env}, {py}, torch {tt.get('torch')}, "
              f"cuda={tt.get('cuda')})", flush=True)
        tmp = Path(f"/tmp/eval_{k}_{os.getpid()}.json")
        rows = []
        err = ""
        # per-series reasons: a model scored on two of three series must explain the
        # empty third table, or a partial sweep reads as a complete one
        series_err: dict[str, str] = {}
        # one subprocess per (model, series): SeriesDataset is single-series, and a
        # per-series pass keeps a partial sweep readable instead of all-or-nothing
        for s in series:
            nm = os.path.splitext(os.path.basename(s))[0]
            cmd = [py, "-m", "tsfm_trl.cli", "evaluate", "--model", k, "--series", s,
                   "--horizons", args.horizons, "--spans", args.spans,
                   "--max-windows", str(args.max_windows),
                   "--device", args.device, "--out", str(tmp),
                   "--ctx", str(args.ctx), "--batch", str(args.batch),
                   "--seed", str(args.seed)]
            env = dict(os.environ)
            env.setdefault("TSFMTRL_QUIET", "1")
            # the child runs `.../env/bin/python` directly, so the venv's own bin dir is
            # not on PATH the way `source activate` would put it: ninja, nvcc shims and
            # other build tools installed *into that env* stay invisible without this
            env["PATH"] = os.path.dirname(py) + os.pathsep + env.get("PATH", "")
            t0 = time.time()
            try:
                p = subprocess.run(cmd, capture_output=True, text=True,
                                   timeout=args.timeout, env=env)
                if tmp.exists() and tmp.stat().st_size:
                    rows.extend(json.loads(tmp.read_text()))
                    tmp.unlink()
                elif p.returncode != 0:
                    tail = (p.stderr or p.stdout or "").strip().splitlines()
                    err = (tail[-1] if tail else f"rc={p.returncode}")[:300]
            except subprocess.TimeoutExpired:
                err = f"timeout after {args.timeout}s"
            except Exception as e:                                          # noqa: BLE001
                err = f"{type(e).__name__}: {e}"[:300]
            if err and not any(r.get("series") == nm for r in rows):
                series_err[nm] = err
            got = len([r for r in rows if r.get("series") == nm])
            print(f"  {k:20s} {nm:14s} "
                  f"{('ok ' + str(got)) if not err else 'FAIL'}"
                  f"  {time.time() - t0:6.1f}s {err[:100]}", flush=True)
        # `interpreter` is provenance, not trivia: the same model measured in a
        # different venv is a different experiment, and the report must be able to say
        # which venv produced which row.
        rec = {"model": k, "env": spec.env, "tier": spec.tier, "family": spec.family,
               "interpreter": py, "device": args.device, "torch": tt,
               "rows": stamp_rows(rows, k), "error": err,
               "series_errors": series_err}
        fh.write(json.dumps(rec, default=float) + "\n")
        fh.flush()
        n_ok += bool(rows)
        n_bad += bool(err and not rows)
    fh.close()
    print(f"swept {n_ok} ok / {n_bad} failed / {skipped} env-skipped -> {out}")
    return 0


def proto(row: dict) -> tuple:
    """What protocol produced this cell. Two measurements with the same signature that
    disagree are run variance; different signatures mean one replaced the other (a context
    contract applied, a device switched, a fixed commit), and the gap between them measures
    the fix, not noise."""
    pr = row.get("_prov") or {}
    return (row.get("ctx"), row.get("ctx_note") or "", pr.get("device"),
            pr.get("code_sha"), str(pr.get("torch")))


def stamp_rows(rows: list, key: str) -> list:
    """Put the registry key on every row.

    An adapter writes its *family* key into a row's `model` field, so `chronos-bolt-small`
    and `chronos-bolt-base` both emit rows named `chronos-bolt`; ten such collisions exist
    in the registry (kronos-base/mini/small all emit `kronos-small`, ttm-r3 emits `ttm-r2`
    ...). Readers that group by `model` then merge different checkpoints and call the
    result a disagreement. The report already stamped this at read time; writing it makes
    the JSONL safe for anything else.
    """
    for r in rows:
        if isinstance(r, dict):
            r.setdefault("model_key", key)
    return rows


def report(args) -> int:
    raw = [Path(x).expanduser() for x in str(args.report).split(",")]
    # "later beats earlier" is meant in time, not alphabetically: retry files sort
    # anywhere, and a stale error-only record must not outlive a newer attempt
    srcs = sorted(raw, key=lambda x: x.stat().st_mtime)
    src = srcs[0]
    # one meta block per input file: a merged report says which machine ran what
    metas: list = []
    by_file = []
    for f in srcs:
        file_recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        fm = next((r["meta"] for r in file_recs if "meta" in r), {})
        metas.append(fm)
        by_file.append((fm, [r for r in file_recs if "meta" not in r]))
    meta = metas[0] if metas else {}

    # Merge at (model, series, span, horizon), not per model. The normal multi-machine
    # case is partial: one machine scored a model on two series, another on the third, a
    # third only produced an error. Merging whole records silently deletes real rows, so
    # later files (by mtime) win *per cell* and each row carries the provenance of the
    # run that produced it.
    cells: dict = {}
    dupes: dict = {}
    err_by_model: dict = {}
    series_err_by_model: dict = {}
    rec_by_model: dict = {}
    for fm, file_recs in by_file:
        for rec in file_recs:
            model = rec["model"]
            prov = {"interpreter": rec.get("interpreter") or "",
                    "device": rec.get("device") or "?", "torch": rec.get("torch"),
                    "machine": fm.get("machine", "?"), "code_sha": fm.get("code_sha"),
                    "started": fm.get("started")}
            for r in rec.get("rows", []):
                r.setdefault("model_key", model)   # registry key, not the adapter's family key
                r.setdefault("dataset", rec.get("dataset"))  # gaps are computed on merged rows
                r["_prov"] = prov
                key = (model, r["series"], r.get("span", "test"), int(r["horizon"]))
                prev = cells.get(key)
                if prev is not None:
                    # A repeated key is not automatically run variance. Most of the 404
                    # duplicates in the pass-3 inputs are a SUPERSEDED protocol: the same
                    # cell scored at ctx 256 before the TTM context contract and at 512
                    # after (MASE 30.1 vs 4.3), which is a fix, not noise. The protocol
                    # signature decides which of the two readings applies.
                    dupes.setdefault(key, []).append(
                        (prev.get("mase"), r.get("mase"),
                         prev["_prov"].get("started"), prov.get("started"),
                         proto(prev), proto(r)))
                cells[key] = r
            if rec.get("error"):
                err_by_model.setdefault(model, rec["error"])
            series_err_by_model.setdefault(model, {}).update(rec.get("series_errors", {}))
            cur = rec_by_model.get(model)
            if cur is None or bool(rec.get("rows")) >= bool(cur.get("rows")):
                rec_by_model[model] = rec
    recs = [rec_by_model[k] for k in sorted(rec_by_model)]
    rows = list(cells.values())
    if dupes:
        # stdout is a data channel: the warning goes to stderr, and the count also lands
        # in the markdown so a reader of the table sees it
        n_same = sum(1 for v in dupes.values() if v[-1][4] == v[-1][5])
        print(f"WARNING: {len(dupes)} cell key(s) appear more than once across the merge "
              f"inputs; the newest file wins per cell. {n_same} under an identical protocol "
              f"(run variance); {len(dupes) - n_same} under a different protocol "
              f"(a superseded measurement - check ctx, device, code_sha).",
              file=sys.stderr)
        for k, v in list(dupes.items())[:5]:
            print(f"  {k} mase {v[-1][0]} -> {v[-1][1]} "
                  f"({'same' if v[-1][4] == v[-1][5] else 'different'} protocol)",
                  file=sys.stderr)
    # which venv and which device produced each row: the same model measured in another
    # interpreter or on another device is a different experiment
    # older merged runs did not stamp their torch build; if that interpreter is on this
    # machine, probe it now rather than shipping a table with unknowns in it
    probe_cmd = ("import json,torch;print(json.dumps({'torch': torch.__version__,"
                 " 'cuda': bool(torch.cuda.is_available())}))")
    seen: dict = {}

    def torch_of(py: str) -> dict:
        if py in seen:
            return seen[py]
        out = {"torch": "?", "cuda": False}
        if py and Path(py).is_absolute() and Path(py).exists():
            try:
                r = subprocess.run([py, "-c", probe_cmd], capture_output=True,
                                   text=True, timeout=120)
                if r.returncode == 0:
                    out = json.loads(r.stdout.strip().splitlines()[-1])
            except Exception:                                  # noqa: BLE001
                pass
        seen[py] = out
        return out

    prov: dict = {}
    for r in rows:
        pr = r.get("_prov") or {}
        k = pr.get("interpreter") or "(interpreter not recorded by that run)"
        # the group key includes machine/device: the same interpreter producing rows on
        # two machines, or on cpu in one run and cuda in another, is two provenance rows
        key = (k, pr.get("machine", "?"), pr.get("device", "?"))
        prov.setdefault(key, {"device": pr.get("device", "?"),
                              "torch": pr.get("torch")
                              or (torch_of(k) if k.startswith("/") else {"torch": "?"}),
                              "machine": pr.get("machine", "?"), "models": set()})
        prov[key]["models"].add(r["model_key"])
    for v in prov.values():
        v["models"] = sorted(v["models"])
    have_rows = {r["model_key"] for r in rows}
    failed = {m: w for m, w in err_by_model.items() if m not in have_rows}
    notrun = [rec["model"] for rec in recs if not rec["rows"] and not rec["error"]]
    series = sorted({r["series"] for r in rows})
    hz = sorted({int(r["horizon"]) for r in rows})
    out = ["# Effectiveness benchmark", "",
           f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} from `{src}` — "
           f"{len(rows)} evaluation rows over {len(series)} series, horizons "
           f"{', '.join(map(str, hz))}. Protocol: `docs/09_benchmark.md`.", "",
           "Rows are **zero-shot** (no adaptation) and slide windows over the span(s) "
           "named above — for zero-shot rows the whole series is legitimate because the "
           "model saw no training data; an *adapted* checkpoint must be scored with "
           "`--spans test`, or the comparison leaks. Same horizon grid, same window "
           "count, same seed for every model. Rows carry the **registry key the sweep "
           "requested** (adapters may report a family-level key internally). `ctx` is the "
           "context actually used (an oversized one is clamped to the span, which is "
           "recorded, not hidden). **CRPS/WQL/pinball appear only for models that emit "
           "quantiles**; a point-only model gets point metrics and is marked `pt` — a "
           "probabilistic column with a dash is not a zero. `skill` is "
           "`1 − MASE` (MASE normalises by the in-sample mean |Δ| of the train span), so "
           "MASE is the column to read and a negative skill means the model lost to that "
           "trivial scaler on this series. `cov@80` is empirical coverage of the nominal "
           "80% band (`nominal_80` in the JSON rows carries the nominal value the model "
           "actually claims, which is not always exactly 0.8 for models with their own "
           "quantile grid).", ""]
    def tcell(t: dict) -> str:
        if t.get("torch") in (None, "?"):
            return "not probed"
        return f"{t.get('torch')} ({'cuda' if t.get('cuda') else 'CPU-only'})"

    if prov:
        out += ["## Which interpreter produced which row", "",
                "Every model runs in the venv its registry env names; a device column is "
                "there because CPU-versus-GPU changes the numbers. `?` means the field "
                "was not recorded by that run, or the interpreter path is relative to a "
                "different machine and cannot be probed from here ('not probed') — treat "
                "those rows as provenance-incomplete.", "",
                "| interpreter | machine | device | torch (cuda?) | models |",
                "|---|---|---|---|---|"] + [
            f"| `{k}` | {v['machine']} | {v['device']} | {tcell(v['torch'])} | "
            f"{', '.join(f'`{m}`' for m in v['models'])} |"
            for (k, _machine, _device), v in sorted(prov.items())] + [""]
    spans = sorted({r.get("span", "test") for r in rows})

    def cell(v):
        return "—" if v is None or (isinstance(v, float) and v != v) else f"{v:.3g}"

    for srs in series:
        for sp in spans:
            for h in hz:
                sel = [r for r in rows if r["series"] == srs
                       and r.get("span", "test") == sp and int(r["horizon"]) == h]
                if not sel:
                    continue
                sel.sort(key=lambda r: (r.get("mase") is None, r.get("mase") or 9e9))
                out += [f"## {srs} · {sp} · horizon {h}", "",
                        "| model | ctx | win | MASE ↓ | RMSE | MAE | dir-acc | rank ρ | "
                        "skill | CRPS ↓ | WQL | pinball | cov@80 |",
                        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
                        "---:|"]
                for r in sel:
                    pt = "" if "crps" in r else " · pt"
                    out.append(f"| {r.get('model_key') or r['model']} | {r['ctx']} | "
                               f"{r['n_windows']} | "
                               f"{cell(r.get('mase'))} | {cell(r.get('rmse'))} | "
                               f"{cell(r.get('mae'))} | {cell(r.get('dir_acc'))} | "
                               f"{cell(r.get('rank_corr'))} | "
                               f"{cell(r.get('relative_skill'))} | "
                               f"{cell(r.get('crps'))}{pt} | {cell(r.get('wql'))} | "
                               f"{cell(r.get('pinball'))} | {cell(r.get('coverage_80'))} |")
                scored = {r["model_key"] for r in rows if r["series"] == srs}
                gaps = sorted((m, errs[srs]) for m, errs in series_err_by_model.items()
                              if errs.get(srs) and m not in scored)
                if gaps:
                    out += ["", "_attempted on this series, not scored:_", ""] + [
                        f"- `{m}`: {w[:150]}" for m, w in gaps]
                out.append("")

    mismatch = [(r["model"], r["env"], r.get("interpreter", "")) for r in recs
                if r["rows"] and r.get("interpreter")
                and f"envs/{r['env']}/" not in r["interpreter"].replace("/home/", "")
                and not r["interpreter"].endswith(f"/{r['env']}/bin/python")
                and f".venv-{r['env']}" not in r["interpreter"] and r["env"] != "core"
                and ".venv2" not in r["interpreter"]]
    if mismatch:
        out += ["## Environment provenance to check", "",
                "Rows whose interpreter does not obviously match the env the registry "
                "names for that model (usually a venv alias — verify before quoting):",
                ""] + [f"- {m} (env {e}) ran with `{i}`" for m, e, i in mismatch] + [""]
    if dupes:
        spread, superseded, by_model, sup_gap = [], [], {}, []
        for (model, _series, _span, _h), seen in dupes.items():
            a, b, _t0, _t1, p0, p1 = seen[-1]    # the pair the merge actually resolved
            if p0 != p1:
                superseded.append(model)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    sup_gap.append(abs(a - b) / max(abs(b), 1e-9))
                continue
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                rel = abs(a - b) / max(abs(b), 1e-9)
                spread.append(rel)
                by_model.setdefault(model, []).append(rel)
        sec = [f"## Repeated cell keys: {len(dupes)}", "",
               "These `(model, series, span, horizon)` keys were measured more than once in "
               "the merge inputs; the table keeps the newest. **{}** of them sit under a "
               "*different* protocol signature (ctx, ctx_note, device, code_sha, torch): an "
               "earlier measurement superseded by a fix, where the gap is the size of that "
               "fix. The other **{}** share a signature, so their gap is run variance.".format(
                   len(superseded), len(spread))]
        if superseded:
            cnt = collections.Counter(superseded)
            sec += ["", "Superseded by a protocol change: "
                        + ", ".join(f"{m} ({n})" for m, n in cnt.most_common(6))]
            if sup_gap:
                sec += ["", f"Gap between a superseded cell and the one that replaced it: "
                            f"median {100*sorted(sup_gap)[len(sup_gap)//2]:.2f}%, max "
                            f"{100*max(sup_gap):.1f}%. Some of these only differ by a code "
                            f"sha (gap ~0.0002%) and are plain re-runs; the large ones are "
                            f"a protocol fix, which is why the number is quoted and not "
                            f"averaged away."]
        if spread:
            sec += ["", f"Run variance under an identical protocol: median "
                        f"{100*sorted(spread)[len(spread)//2]:.2f}%, max "
                        f"{100*max(spread):.2f}% over {len(spread)} comparable pairs.", "",
                    "Per model (median / max over its repeated pairs) — this is where a "
                    "family's numbers stop being one number:", "",
                    "| model | pairs | median | max |", "|---|---|---|---|"]
            for m, v in sorted(by_model.items(), key=lambda kv: -sorted(kv[1])[len(kv[1]) // 2]):
                sec.append(f"| {m} | {len(v)} | {100*sorted(v)[len(v)//2]:.2f}% "
                           f"| {100*max(v):.2f}% |")
        sec += ["", "Quote a model's ranking only where its gap to the next model exceeds "
                "the run-variance spread above.", ""]
        out += sec
    # a table that leaves models out is read as "they were fine"; say who is missing
    try:
        reg_keys = set(default_registry().keys())
    except Exception:                                          # noqa: BLE001
        reg_keys = set()
    absent = sorted(reg_keys - {r["model"] for r in recs})
    if absent:
        out += [f"## Not in this table ({len(absent)} of {len(reg_keys)} registry models)",
                "", "No row was produced for these yet — they are **not** measured here, "
                "and their absence is not a pass:", "",
                ", ".join(f"`{m}`" for m in absent), ""]
    # Anything that counts *measurements* must read `rows` (the merged cells), not `recs`:
    # `rec_by_model` keeps ONE raw record per model, so with two GPU lanes — or a GPU lane
    # plus a CPU retry — a rec-based count silently reports one lane. That under-counted
    # this section in pass 2 and would have reported 33 of 66 flagged MOMENT series here.
    scored_models = {r.get("model_key") or r.get("model") for r in rows}
    dataset_gaps = [(rec["model"], rec["error"]) for rec in recs
                    if rec.get("error") and "covariate panel" in rec["error"]
                    and (rec.get("model_key") or rec.get("model")) not in scored_models]
    dataset_gaps = sorted(set(dataset_gaps))
    if dataset_gaps:
        out += ["## Not scoreable on some series (dataset coverage, not model failure)", "",
                "These models consume a covariate panel; the close-only series cannot "
                "score them at all, so their absence from a table is about the dataset:",
                ""] + [f"- **{m}**: {w[:160]}" for m, w in dataset_gaps] + [""]
    # rows whose adapter says the task head was never trained for the task. MOMENT's
    # released checkpoint has only a reconstruction head, so its zero-shot forecast row
    # scores a randomly initialised linear map: keep the number, label it.
    untrained = {}
    for r in rows:
        mt = r.get("meta") or {}
        if mt.get("head_from_checkpoint") is False and not mt.get("head_trained_here"):
            untrained.setdefault(r.get("model_key") or r.get("model"), set()).add(
                str(r.get("series", "?")))
    if untrained:
        out += ["## Flagged: the scored head was never trained", "",
                "These rows ran fine and are **kept**, because a number you can see beats a "
                "blank — but they do not measure the model doing this task. The checkpoint "
                "ships no trained head for it, and `init()` randomises the one that gets "
                "built. Read them as a reference point for a fine-tune column, not as "
                "zero-shot performance:", ""] + [
            f"- **{m}**: {len(s)} series ("
            + ", ".join(sorted(s)[:6]) + (" …" if len(s) > 6 else "") + ")"
            for m, s in sorted(untrained.items())] + [""]
    if failed:
        out += ["## Failed or incomplete rows", ""]
        for m, why in sorted(failed.items()):
            out.append(f"- **{m}**: {why}")
        out.append("")
    if notrun:
        out += [f"## No rows (attempted, nothing scored): {', '.join(notrun)}", ""]
    text = "\n".join(out) + "\n"
    if args.md:
        # a report path names a run directory that does not exist until this line; making
        # it here means --md runs/new-dir/report.md is not a footgun on first use
        Path(args.md).expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text(text)
        print(f"wrote {args.md}")
    if args.stdout:
        sys.stdout.write(text)
    return 0


def _abspath(py: str) -> str:
    """Absolute, but lexically: `.venv/bin/python` is a symlink to the base
    interpreter, and Path.resolve() follows it out of the venv — the resulting path
    starts in uv's python dir, has no site-packages, and dies on `import numpy` while
    looking like a broken model environment. abspath() keeps the venv identity."""
    return os.path.abspath(os.path.expanduser(py))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--series", action="append", default=[],
                    help="CSV with timestamp,close (repeatable)")
    ap.add_argument("--horizons", default="21,63,126")
    ap.add_argument("--max-windows", type=int, default=16)
    # protocol knobs the child inherits, so the file states what was actually asked for
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spans", default="all",
                    help="zero-shot rows slide over 'all'; use 'test' when scoring "
                         "adapted checkpoints, or the comparison leaks")
    ap.add_argument("--models", default="all")
    ap.add_argument("--tier", default="", choices=["", "A", "B", "C"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--envs", default="envs")
    ap.add_argument("--allow-core-fallback", action="store_true",
                    help="run a model in the core interpreter if its own env is missing "
                         "(off by default: numbers from an environment the matrix does "
                         "not name for that model are a different experiment)")
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--out", default="reports/effectiveness.jsonl")
    ap.add_argument("--fresh", action="store_true", help="truncate the JSONL first")
    ap.add_argument("--report", default="",
                    help="render markdown from JSONL(s); comma-separated merges several "
                         "runs/machines (rows beat errors)")
    ap.add_argument("--md", default="EFFECTIVENESS.md")
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()
    if args.report:
        return report(args)
    if not args.series:
        raise SystemExit("--series is required for a sweep (a benchmark needs a named "
                         "dataset; the built-in fixture is synthetic and only proves plumbing)")
    return sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
