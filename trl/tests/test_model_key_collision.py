"""A row's `model` is the adapter's family key, not the registry key.

Ten registry keys share a row-level id: `chronos-bolt-small` and `chronos-bolt-base` both
emit `chronos-bolt`, `kronos-base`/`-mini`/`-small` all emit `kronos-small`, `ttm-r3`
emits `ttm-r2`, `moment-1-small` emits `moment-1-large`. A comparison or ranking that
groups rows by that field merges different checkpoints — which is how a "3-40% cross-
machine disagreement" appeared in this session's notes before it was traced to the key.
The registry key must ride on the row.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("sweep_evaluate",
                                                  HERE / "scripts" / "sweep_evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stamp_rows_puts_the_registry_key_on_every_row():
    mod = _load()
    rows = [{"model": "chronos-bolt", "series": "s1", "horizon": 21, "mase": 2.2},
            {"model": "chronos-bolt", "series": "s2", "horizon": 21, "mase": 3.1},
            {"model": "chronos-bolt", "series": "s3", "horizon": 21, "model_key": "keep"}]
    out = mod.stamp_rows(rows, "chronos-bolt-base")
    assert [r["model_key"] for r in out[:2]] == ["chronos-bolt-base"] * 2
    assert out[2]["model_key"] == "keep", "an existing key is a fact, not a default to crush"


def test_two_checkpoints_sharing_a_family_key_stay_two_cells(tmp_path):
    """The merge must not call chronos-bolt-small vs -base a duplicated cell."""
    mod = _load()

    def rec(key, mase):
        return {"model": key, "env": "core", "tier": "A", "family": "chronos_bolt",
                "interpreter": "/x/core/bin/python", "device": "cuda",
                "torch": {"torch": "2.7.1+cu128"},
                "rows": [mod.stamp_rows([{"model": "chronos-bolt", "series": "s1",
                                          "span": "all", "horizon": 21, "ctx": 256,
                                          "n_windows": 8, "mase": mase,
                                          "no_leakage": True}], key)[0]]}

    f = tmp_path / "lane.jsonl"
    f.write_text("\n".join([
        json.dumps({"meta": {"started": "2026-01-01 00:00:00", "machine": "trux",
                             "code_sha": "abc", "python": "3.11.0", "device": "cuda",
                             "ctx": 256, "horizons": [21], "spans": "all",
                             "max_windows": 8, "batch": 16, "seed": 0}}),
        json.dumps(rec("chronos-bolt-small", 2.2547)),
        json.dumps(rec("chronos-bolt-base", 2.3330)),
    ]) + "\n")
    md = tmp_path / "out.md"
    mod.report(argparse.Namespace(report=str(f), md=str(md), stdout=False))
    text = md.read_text()
    assert "Repeated cell keys" not in text, "two checkpoints are not one cell measured twice"
    assert "chronos-bolt-small" in text and "chronos-bolt-base" in text
