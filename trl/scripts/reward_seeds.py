#!/usr/bin/env python3
"""Repeated reward runs for one model: the honest version of "reward helped".

A single reward curve on a few dozen windows cannot support a claim — the quartile
means in the matrix say "the late quarter beat the early quarter", not "improvement".
This runs the same command over several seeds and reports the spread of the
late-minus-early reward delta, so a positive result either survives repetition or
does not.

    scripts/reward_seeds.py --model chronos-t5-small --seeds 0,1,2 \
        --steps 96 --bs 8 --lr 2e-5 --budget fs64 --peft lora --device cuda
"""
from __future__ import annotations

import argparse
import json
import socket
import statistics
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="")
    ap.add_argument("--fresh", action="store_true",
                    help="overwrite an existing --out (different configs must not clobber "
                         "each other's evidence by accident)")
    ap.add_argument("--python", default=sys.executable)
    for flag, kind in (("--steps", int), ("--bs", int), ("--lr", float), ("--ctx", int),
                       ("--horizon", int), ("--max-windows", int), ("--kl", float),
                       ("--candidates", int), ("--pinball-weight", float)):
        ap.add_argument(flag, type=kind, default=None)
    for flag in ("--budget", "--peft", "--device", "--mode", "--series"):
        ap.add_argument(flag, default=None)
    a = ap.parse_args()

    cmd_extra: list[str] = []
    # argparse stores dashes as underscores; map the flags explicitly rather than
    # deriving the attribute name from the flag
    names = {"--steps": "steps", "--bs": "bs", "--lr": "lr", "--ctx": "ctx",
             "--horizon": "horizon", "--max-windows": "max_windows", "--kl": "kl",
             "--candidates": "candidates", "--pinball-weight": "pinball_weight",
             "--budget": "budget", "--peft": "peft", "--device": "device",
             "--mode": "mode", "--series": "series"}
    for flag, attr in names.items():
        v = getattr(a, attr)
        if v is not None:
            cmd_extra += [flag, str(v)]

    runs = []
    for seed in [int(s) for s in a.seeds.split(",") if s.strip()]:
        cmd = [a.python, "-m", "tsfm_trl.cli", "reward", "--model", a.model,
               "--seed", str(seed)] + cmd_extra
        p = subprocess.run(cmd, capture_output=True, text=True)
        try:
            row = json.loads(p.stdout)
        except json.JSONDecodeError:
            tail = (p.stderr or p.stdout or "").strip().splitlines()
            row = {"status": "failed", "reason": (tail[-1] if tail else "no output")[:200]}
        row["seed"] = seed
        runs.append(row)
        e, l = row.get("reward_early_mean"), row.get("reward_late_mean")
        print(f"  seed {seed}: mode={row.get('mode')} status={row.get('status')} "
              f"early={e} late={l} delta="
              f"{(round(l - e, 4) if isinstance(e, (int, float)) and isinstance(l, (int, float)) else None)} "
              f"trainable={row.get('trainable_parameters')} {(row.get('reason') or '')[:80]}",
              flush=True)

    deltas = [r["reward_late_mean"] - r["reward_early_mean"] for r in runs
              if isinstance(r.get("reward_early_mean"), (int, float))
              and isinstance(r.get("reward_late_mean"), (int, float))]
    # a reward curve without its configuration is not evidence: the same model+mode
    # flips sign across learning rates, and two machines writing the same filename
    # silently overwrite each other's run
    try:
        sha = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent.parent),
                              "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:                                          # noqa: BLE001
        sha = "?"
    probe = ("import json,torch;print(json.dumps({'torch': torch.__version__,"
             " 'cuda': bool(torch.cuda.is_available())}))")
    try:
        pr = subprocess.run([a.python, "-c", probe], capture_output=True, text=True,
                            timeout=120)
        child_torch = json.loads(pr.stdout.strip().splitlines()[-1]) if pr.returncode == 0 \
            else {"torch": "?", "cuda": False}
    except Exception:                                          # noqa: BLE001
        child_torch = {"torch": "?", "cuda": False}
    if (a.__dict__.get("device") or "").startswith("cuda") and not child_torch.get("cuda"):
        print(f"  NOTE: --device cuda requested but {a.python} reports no CUDA "
              f"(torch {child_torch.get('torch')}) — the run will fail or fall back")
    summary = {"model": a.model, "seeds": len(runs), "runs": runs,
               "protocol": {"machine": socket.gethostname(), "code_sha": sha,
                            "python": sys.version.split()[0], "argv": sys.argv[1:],
                            "child_python": a.python, "child_torch": child_torch},
               "delta_mean": round(statistics.mean(deltas), 5) if deltas else None,
               "delta_sd": round(statistics.stdev(deltas), 5) if len(deltas) > 1 else None,
               "n_improved": sum(1 for d in deltas if d > 0),
               "modes": sorted({str(r.get("mode")) for r in runs})}
    if deltas:
        spread = f"{summary['delta_mean']:+.4f}"
        if summary["delta_sd"] is not None:
            spread += f" ± {summary['delta_sd']:.4f}"
        summary["verdict"] = (f"late-minus-early reward {spread} over {len(deltas)} seeds; "
                              f"{summary['n_improved']}/{len(deltas)} seeds improved")
        print(summary["verdict"])
    else:
        summary["verdict"] = "no seed produced reward quartile means (see reasons)"
        print(summary["verdict"])
    if a.out:
        p = Path(a.out)
        if p.exists() and not a.fresh:
            raise SystemExit(f"{p} already exists — pass --fresh or choose another "
                             f"--out; different configurations must not overwrite "
                             f"each other's evidence")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, indent=1, default=float))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
