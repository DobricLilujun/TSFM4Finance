"""A row must be able to survive being alone.

Lanes get filtered, split, uploaded, and merged by hand. When a row is separated from the
meta line at the head of its file - a jq filter, a lane truncated by a resume, a report that
reads one record - the machine and commit that produced it vanish, and the number becomes
unattributable. The lane header is still written (it is the cheaper provenance for tables),
but every row now carries `machine` and `code_sha` too.
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SCRIPT = HERE / "scripts" / "sweep_evaluate.py"


def test_a_lane_written_by_the_sweep_puts_machine_and_sha_on_every_row(tmp_path):
    # one real cell, no weights: the naive baseline is enough to exercise the write path
    import numpy as np
    import pandas as pd
    v = np.cumsum(np.random.default_rng(1).normal(0, 1, 400)) + 50.0
    df = pd.DataFrame({"timestamp": pd.date_range("2000-01-01", periods=len(v), freq="D"),
                       "close": v})
    csv = tmp_path / "tiny.csv"
    df.to_csv(csv, index=False)
    envs = tmp_path / "envs"
    envs.mkdir()
    (envs / "local.json").write_text(json.dumps({"core": sys.executable}))

    out = tmp_path / "lane.jsonl"
    p = subprocess.run(
        [sys.executable, str(SCRIPT), "--series", str(csv), "--models", "naive",
         "--horizons", "8", "--max-windows", "2", "--ctx", "64", "--seed", "0",
         "--device", "cpu", "--envs", str(envs), "--out", str(out), "--fresh"],
        capture_output=True, text=True, timeout=300, cwd=str(HERE))
    assert p.returncode == 0, p.stderr[-800:]

    recs = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    meta = next(r["meta"] for r in recs if "meta" in r)
    rows = [row for r in recs if "meta" not in r for row in (r.get("rows") or [])]
    assert rows, "the sweep produced no scored rows"
    for row in rows:
        assert row["machine"] == meta["machine"] == socket.gethostname()
        assert row["code_sha"] == meta["code_sha"]
        assert row["model_key"] == "naive"


def test_stamp_rows_never_overwrites_what_the_adapter_already_said():
    from test_merge_sections import _load  # same loader: scripts/ is not a package
    mod = _load()
    rows = [{"model": "chronos-bolt", "model_key": "chronos-bolt-small", "machine": "gpu7"},
            {"model": "chronos-bolt"}]
    mod.stamp_rows(rows, "chronos-bolt-base", {"machine": "trux-tsubame", "code_sha": "abc123"})
    assert rows[0]["model_key"] == "chronos-bolt-small"
    assert rows[0]["machine"] == "gpu7"           # already stamped: left alone
    assert rows[1]["model_key"] == "chronos-bolt-base"
    assert rows[1]["machine"] == "trux-tsubame"
    assert rows[1]["code_sha"] == "abc123"
