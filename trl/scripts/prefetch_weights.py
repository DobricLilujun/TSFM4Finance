#!/usr/bin/env python3
"""Download every registry model's weights once, into $TSFM_WEIGHTS_ROOT.

The matrix is only reproducible if the weights are on disk *before* it runs: a row that
silently downloads 5 GB mid-measurement both skews the timings and hides an offline
failure as a "model bug".

    TSFM_WEIGHTS_ROOT=$HOME/models ./scripts/prefetch_weights.py --tier A
    ./scripts/prefetch_weights.py --env moirai --dry-run

Rows whose weights live on the authors' own cloud (VisionTS++, the t0 family) are listed
as `weights: null` in models.yaml and are skipped with a pointer, not fetched.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tsfm_trl.model.registry import default_registry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="", help="comma list, e.g. A,B")
    ap.add_argument("--env", default="", help="only this env (core, moirai, ...)")
    ap.add_argument("--model", default="", help="comma list of registry keys")
    ap.add_argument("--root", default=os.environ.get("TSFM_WEIGHTS_ROOT", ""),
                    help="target root (default $TSFM_WEIGHTS_ROOT)")
    ap.add_argument("--pattern", action="append", default=[],
                    help="allow_patterns, repeatable (e.g. '*.safetensors' to skip optimizer state)")
    ap.add_argument("--only-missing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    reg = default_registry()
    specs = [s for s in reg.models()
             if (not a.tier or s.tier in a.tier.split(","))
             and (not a.env or s.env == a.env)
             and (not a.model or s.key in a.model.split(","))]
    root = Path(a.root).expanduser() if a.root else None
    from huggingface_hub import snapshot_download

    ok = miss = skip = fail = 0
    for s in specs:
        if not s.weights:
            print(f"--  {s.key}: no public HF weights "
                  f"({s.notes or s.install or 'see docs/08_known_issues.md'})")
            skip += 1
            continue
        dest = None
        if root:
            # registry ids are 'Org/Name'; mirror that layout so the weights resolver
            # (which matches by trailing name) can find them without an HF cache scan
            dest = root / s.weights
        if a.only_missing and dest and dest.is_dir() and any(dest.glob("*.safetensors")):
            print(f"--  {s.key}: present at {dest}")
            miss += 1
            continue
        print(f"->  {s.key}: {s.weights}" + (f" -> {dest}" if dest else " (HF cache)"))
        if a.dry_run:
            continue
        try:
            snapshot_download(repo_id=s.weights, local_dir=str(dest) if dest else None,
                              allow_patterns=a.pattern or None)
            ok += 1
        except Exception as e:                                        # noqa: BLE001
            print(f"    FAILED {type(e).__name__}: {str(e)[:200]}")
            fail += 1
    print(f"\n{len(specs)} rows: {ok} downloaded, {miss} already present, "
          f"{skip} no public weights, {fail} failed")
    return 0 if not fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
