#!/usr/bin/env python3
"""trux lane runner: split an effectiveness sweep across the two shared GPUs + CPU.

This is a scheduler, not a reimplementation: all scoring still happens in the harness's
`scripts/sweep_evaluate.py`, which already runs one subprocess per (model, series) in that
model's own family environment, appends incrementally to JSONL, and merges per
(model, series, span, horizon). Here we only decide *which series go to which device*,
*when it is safe to touch a GPU at all*, and *what the machine looked like while it ran*.

Why each piece exists (all of these were observed on trux, not hypothesised):

* The two RTX 6000 Ada cards are shared with other users — during this session they held
  17-21 GB and ran at 96-100% utilisation. So lanes are assigned by **currently free**
  memory, not by "there are two cards, use both".
* A family env can silently resolve a CPU-only torch (seen on DGX: `2.7.0+cpu`), and
  `--device cuda` then produces rows nobody asked for. The runner refuses to start a GPU
  lane whose env cannot see CUDA, instead of scoring into a silently-CPU row.
* A slow row on a contended card is not a model property, so every series records a
  contention snapshot (free MiB + who holds the rest) next to the results.
* Interrupted runs must not lose work: already-scored series are skipped on restart.

`--dry-run` prints the exact commands and needs no repo, which is how the splitting and
device-guard logic can be checked before the code has even arrived.

Usage:
    trux_lane_runner.py --series-file sweep_stratified.txt --dry-run
    trux_lane_runner.py --series-file sweep_stratified.txt --horizons 21,63,126 \
        --windows 12 --out-dir reports/trux_pass2
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

UV_ENV_ROOT = "/raid/yewei/envs/tsfm-trl"


def sh(cmd: list[str], timeout: int = 60) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout).stdout.strip()
    except Exception as e:                                   # noqa: BLE001
        return f"<error {type(e).__name__}: {e}>"


def gpu_free() -> list[dict]:
    """Physical GPU index -> free/total MiB. None if nvidia-smi is missing."""
    if not shutil.which("nvidia-smi"):
        return []
    out = sh(["nvidia-smi", "--query-gpu=index,memory.free,memory.total",
              "--format=csv,noheader,nounits"])
    gpus = []
    for line in out.splitlines():
        try:
            i, free, total = (x.strip() for x in line.split(","))
            gpus.append({"index": int(i), "free_mib": int(free), "total_mib": int(total)})
        except ValueError:
            continue
    return gpus


def gpu_context() -> dict:
    """Who else is on the card right now — the context a timing number needs."""
    apps = []
    if shutil.which("nvidia-smi"):
        out = sh(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                  "--format=csv,noheader"])
        for line in out.splitlines():
            pid, _, mem = line.partition(",")
            pid = pid.strip()
            user = sh(["ps", "-o", "user=", "-p", pid]) or "?"
            age = sh(["ps", "-o", "etimes=", "-p", pid]) or "?"
            apps.append({"pid": pid, "user": user.strip(), "mem": mem.strip(),
                         "age_s": age.strip()})
    return {"free": gpu_free(), "other_processes": apps,
            "loadavg": " ".join(Path("/proc/loadavg").read_text().split()[:3])
            if Path("/proc/loadavg").exists() else "?"}


def env_has_cuda(env_dir: Path) -> tuple[bool, str]:
    py = env_dir / "bin" / "python"
    if not py.exists():
        return False, f"no interpreter at {py}"
    out = sh([str(py), "-c",
              "import torch;print(torch.__version__, torch.cuda.is_available())"],
             timeout=180)
    return ("True" in out), out or "no output"


def plan_lanes(series: list[tuple[str, int]], lanes: list[str]) -> list[list[str]]:
    """Assign series to lanes by a cost proxy, longest first, lightest lane next.

    The proxy is the number of test rows: the harness slices a fixed number of windows
    per series, but hourly series are ~20x longer and dominate input/context handling.
    Once a real run has produced measured seconds per series, replace this function's
    weight with the measured cost — that is the point of keeping it isolated.
    """
    buckets: list[list[float]] = [[] for _ in lanes]  # parallel list of weights
    out: list[list[str]] = [[] for _ in lanes]
    for path, weight in sorted(series, key=lambda x: -x[1]):
        k = min(range(len(lanes)), key=lambda i: sum(buckets[i]))
        out[k].append(path)
        buckets[k].append(float(weight))
    return out


def already_scored(jsonl: Path) -> set[str]:
    """Series names with at least one ok row, so a restart does not redo them."""
    done: set[str] = set()
    if not jsonl.exists():
        return done
    for line in jsonl.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("series"):
            done.add(str(r["series"]))
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/raid/yewei/projects_yewei/tsfm-trl")
    ap.add_argument("--env-root", default=UV_ENV_ROOT)
    ap.add_argument("--series-file", required=True)
    ap.add_argument("--out-dir", default="reports/trux_pass2")
    ap.add_argument("--lanes", default="cuda-a,cuda-b,cpu",
                    help="lane names; devices are resolved by free memory at start")
    ap.add_argument("--horizons", default="21,63,126")
    ap.add_argument("--windows", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep-arg", action="append", default=[],
                    help="extra argument passed verbatim to sweep_evaluate.py "
                         "(repeatable); verify spellings against --help")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--min-free-mib", type=int, default=6000,
                    help="a GPU lane needs at least this much free memory on its card; "
                         "trux cards are shared and a card can drop to <1 GiB free "
                         "between measurements")
    a = ap.parse_args()

    series_paths = [l.strip() for l in Path(a.series_file).read_text().splitlines()
                    if l.strip() and not l.startswith("#")]
    missing = [p for p in series_paths if not Path(p).is_dir()]
    if missing:
        print(f"ERROR: {len(missing)} series dirs missing, e.g. {missing[:3]}",
              file=sys.stderr)
        return 2

    # cost proxy; see plan_lanes
    weighted = []
    for p in series_paths:
        try:
            weighted.append((p, max((Path(p) / "train.parquet").stat().st_size // 40, 1)))
        except OSError:
            weighted.append((p, 1000))

    gpus = gpu_free()
    lanes = [x.strip() for x in a.lanes.split(",") if x.strip()]
    assign: dict[str, dict] = {}
    gpu_pool = sorted([g for g in gpus if g["free_mib"] >= a.min_free_mib],
                      key=lambda g: -g["free_mib"])          # roomiest usable card first
    for name in lanes:
        if name.startswith("cuda") or name.startswith("gpu"):
            if not gpu_pool:
                # not an error: run it on CPU with the device recorded as cpu. Launching
                # a GPU lane onto a 644 MiB card just produces a failed sweep.
                print(f"WARNING: no card with >= {a.min_free_mib} MiB free "
                      f"(free: {[g['free_mib'] for g in gpus]}); lane {name!r} runs on CPU "
                      f"and its rows are labelled device=cpu",
                      file=sys.stderr)
                assign[name] = {"device": "cpu", "visible": "", "free_mib": None,
                                "downgraded_from_gpu": True}
                continue
            g = gpu_pool.pop(0)
            assign[name] = {"device": "cuda", "visible": str(g["index"]),
                            "free_mib": g["free_mib"]}
        else:
            assign[name] = {"device": "cpu", "visible": "", "free_mib": None}

    print(f"GPUs now: {json.dumps(gpus)}")
    for n, cfg in assign.items():
        print(f"lane {n:8s} -> device={cfg['device']:4s} "
              f"CUDA_VISIBLE_DEVICES={cfg['visible']!r:5} "
              f"card_free={cfg['free_mib']} MiB")

    buckets = plan_lanes(weighted, lanes)
    for n, items in zip(lanes, buckets):
        print(f"lane {n:8s} {len(items)} series")

    sweep = Path(a.repo) / "scripts" / "sweep_evaluate.py"
    env_core = Path(a.env_root) / "core"
    ok_cuda, probe = (True, "not probed (dry run)")
    if not a.dry_run:
        if not sweep.exists():
            print(f"ERROR: {sweep} not found — repo not cloned here yet", file=sys.stderr)
            return 2
        ok_cuda, probe = env_has_cuda(env_core)
        print(f"core env torch probe: {probe}")
        gpu_lanes = [n for n, c in assign.items() if c["device"] == "cuda"]
        if gpu_lanes and not ok_cuda:
            print(f"ERROR: GPU lanes {gpu_lanes} requested but {env_core} cannot see "
                  f"CUDA ({probe}). Refusing to produce silently-CPU rows.",
                  file=sys.stderr)
            return 3

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plan.json").write_text(json.dumps({
        "when": dt.datetime.now(dt.timezone.utc).isoformat(),
        "series_file": str(a.series_file), "series": len(series_paths),
        "horizons": a.horizons, "windows": a.windows, "seed": a.seed,
        "lanes": assign, "buckets": {n: b for n, b in zip(lanes, buckets)},
        "gpus_at_plan": gpus, "core_env_torch": probe,
        "min_free_mib": a.min_free_mib,
        "cost_proxy": "train.parquet size / 4 (replace with measured seconds)"},
        indent=1))

    procs = []
    for name, items in zip(lanes, buckets):
        jsonl = out_dir / f"lane_{name}.jsonl"
        done = already_scored(jsonl)
        # the harness may record the series name or its path; accept either so a
        # restart never re-scores a series it already measured
        todo = [p for p in items if Path(p).name not in done and p not in done]
        if done:
            print(f"lane {name:8s} resume: {len(done)} series already scored, "
                  f"{len(todo)} to go")
        env = dict(os.environ)
        if assign[name]["device"] == "cuda":
            env["CUDA_VISIBLE_DEVICES"] = assign[name]["visible"]
        else:
            env.pop("CUDA_VISIBLE_DEVICES", None)
            env["OMP_NUM_THREADS"] = "16"
        cmd = [str(env_core / "bin" / "python"), str(sweep),
               "--horizons", a.horizons, "--windows", str(a.windows),
               "--seed", str(a.seed), "--device", assign[name]["device"],
               "--out", str(jsonl)] + list(a.sweep_arg) + [str(p) for p in todo]
        if a.dry_run:
            print(f"  [{name}] CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')!r} "
                  + " ".join(cmd[:6]) + f" ... {len(todo)} series args")
            continue
        if not todo:
            continue
        log = out_dir / f"lane_{name}.log"
        procs.append((name, jsonl, subprocess.Popen(cmd, env=env, stdout=log.open("w"),
                                                    stderr=subprocess.STDOUT)))
        # a snapshot per lane start so a slow row can be attributed, not argued about
        (out_dir / f"lane_{name}.env.jsonl").write_text(
            json.dumps({"at": dt.datetime.now(dt.timezone.utc).isoformat(),
                        **gpu_context()}) + "\n")

    if a.dry_run:
        print("dry run: nothing launched (plan written to "
              f"{out_dir / 'plan.json'})")
        return 0
    if not procs:
        print("nothing to do")
        return 0

    t0 = time.time()
    for name, jsonl, p in procs:
        rc = p.wait()
        snap = gpu_context()
        with (out_dir / f"lane_{name}.env.jsonl").open("a") as f:
            f.write(json.dumps({"at": dt.datetime.now(dt.timezone.utc).isoformat(),
                                "rc": rc, "wall_s": round(time.time() - t0, 1),
                                **snap}) + "\n")
        rows = len(jsonl.read_text().splitlines()) if jsonl.exists() else 0
        print(f"lane {name}: rc={rc} rows={rows}")
    print(f"total wall {time.time() - t0:.0f}s; merge with the harness's own "
          f"--report/--md over {out_dir}/lane_*.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
