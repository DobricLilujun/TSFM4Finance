"""The window-sensitivity probe must group by `n_windows` and never merge across it.

A 32-window row and an 8-window row answer different questions about the same series, so the probe
that compares them keeps them in separate groups and reports the difference. If it merged them the
way the report merge does (newest wins), the whole point - that the ranking is a function of the
window draw - would disappear into a single table.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SCRIPT = HERE / "probes" / "window_sensitivity.py"


def _lane(path, n_windows, cells, machine="sedan-gpu3", data_hash="abc123"):
    """cells: model -> {series -> (mase_model, mase_naive)} for h=21."""
    recs = [{"meta": {"machine": machine, "max_windows": n_windows, "code_sha": "deadbeef"}}]
    rows = []
    for m, per in cells.items():
        for s, (mm, mn) in per.items():
            rows.append({"model": m, "model_key": m, "series": s, "span": "all", "horizon": 21,
                         "mase": mm, "n_windows": n_windows, "machine": machine,
                         "data_hash": data_hash, "data_kind": "arena_dir"})
    recs.append({"model": "the-lane", "rows": rows})
    Path(path).write_text("\n".join(json.dumps(r) for r in recs) + "\n")


def test_two_window_counts_stay_two_groups_and_the_flips_are_named(tmp_path):
    # flip: "switch" beats naive on 8 windows and loses on 32; "winner" wins both;
    # "onlybig" exists only in the 32-window lane
    _lane(tmp_path / "l8.jsonl", 8, {
        "naive": {"s1": (2.0, 2.0), "s2": (2.0, 2.0)},
        "switch": {"s1": (1.8, 2.0), "s2": (1.9, 2.0)},
        "winner": {"s1": (1.5, 2.0), "s2": (1.4, 2.0)},
    })
    _lane(tmp_path / "l32.jsonl", 32, {
        "naive": {"s1": (2.0, 2.0), "s2": (2.0, 2.0)},
        "switch": {"s1": (2.4, 2.0), "s2": (2.2, 2.0)},
        "winner": {"s1": (1.6, 2.0), "s2": (1.7, 2.0)},
        "onlybig": {"s1": (1.1, 2.0), "s2": (1.2, 2.0)},
    })
    p = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path / "l8.jsonl"),
                        str(tmp_path / "l32.jsonl")],
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-600:]
    out = p.stdout
    assert "8 windows:" in out and "32 windows:" in out
    # shares are reported per group; nothing is merged
    assert "switch" in out and "onlybig" in out
    flips = out.split("cell-level outcome flips")[1]
    assert "cell-level outcome flips" in out
    # the table is printed as "model  f/ n cells change winner" with the denominator right-aligned
    got = {m.group(1): (int(m.group(2)), int(m.group(3)))
           for m in re.finditer(r"^\s+(\S+)\s+(\d+)/\s*(\d+)", flips, flags=re.M)}
    assert got.get("switch") == (2, 2), f"the flipping cells are not counted: {got}"
    assert got.get("winner") == (0, 2), f"a stable cell must not count as a flip: {got}"
    assert got.get("ALL") == (2, 4), got
    assert "32.3" not in out            # the fixture is small; no real-world numbers leak in
    assert "was never a measurement of model quality" in out


def test_cells_without_the_window_field_are_called_out_not_silently_grouped(tmp_path):
    old = tmp_path / "old.jsonl"
    _lane(old, 8, {"naive": {"s1": (2.0, 2.0)}, "a": {"s1": (1.9, 2.0)}})
    # strip n_windows from the old lane: it predates the field
    recs = [json.loads(l) for l in old.read_text().splitlines()]
    for r in recs:
        for row in r.get("rows", []):
            row.pop("n_windows")
    old.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    _lane(tmp_path / "new.jsonl", 32, {"naive": {"s1": (2.0, 2.0)}, "a": {"s1": (2.1, 2.0)}})
    p = subprocess.run([sys.executable, str(SCRIPT), str(old), str(tmp_path / "new.jsonl")],
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-600:]
    assert "predate the n_windows field" in p.stdout
