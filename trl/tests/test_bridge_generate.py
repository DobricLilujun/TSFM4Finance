"""The bridge generator has to inline *everything*, and prove it.

Two bridge jobs died the same way for different reasons: job_012 because a guard inside a generated
file compared against the placeholder the generator substitutes (so the check was `[ x != x ]` and
passed on an ungenerated file), job_014 because TRUX/TRUX_SSH/SRC were read from a bridge.env that
exists on trux and nowhere on the worker. Both are the same rule - one file crosses the bridge - and
both should fail in a test rather than on a far machine at 03:00.
"""
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GEN = REPO / "probes" / "trux" / "bridge" / "gpu7_generate.sh"

TEMPLATE = """#!/usr/bin/env bash
set -euo pipefail
MARKER='@@'
TRUX=${TRUX:-"@@TRUX@@"}
SERIES_LIST=${SERIES_LIST:-"@@SERIES@@"}
case " $SERIES_LIST " in *"$MARKER"*) echo "FATAL: marker left"; exit 6 ;; esac
echo "TRUX=$TRUX"
echo "SERIES=$SERIES_LIST"
"""


def _bridge(tmp_path, **vals):
    env = tmp_path / "bridge.env"
    env.write_text("".join(f'{k}="{v}"\n' for k, v in vals.items()))
    return env


def _run(tmp_path, tpl, series_file):
    # the script reads bridge.env from its own directory, so the test ships a copy
    script = tmp_path / "gpu7_generate.sh"
    script.write_text(GEN.read_text())
    return subprocess.run(["bash", str(script), str(tpl), str(series_file),
                           str(tmp_path / "job.sh")], capture_output=True, text=True)


def test_generated_job_carries_the_series_and_the_endpoints(tmp_path):
    tpl = tmp_path / "tpl.sh"
    tpl.write_text(TEMPLATE)
    ser = tmp_path / "series.txt"
    ser.write_text("std_equity_1d_10107\nstd_fx_1d_eurjpy\n")
    _bridge(tmp_path, BRIDGE="/b", TRUX="trux.example", TRUX_SSH="ssh -p 22 -i /k",
            SRC="/data/standard")
    r = _run(tmp_path, tpl, ser)
    assert r.returncode == 0, r.stderr
    job = (tmp_path / "job.sh").read_text()
    assert "@@" not in job.replace("MARKER='@@'", ""), job
    assert "std_equity_1d_10107 std_fx_1d_eurjpy" in job
    assert 'TRUX=${TRUX:-"trux.example"}' in job
    # and the guard inside the generated file still refuses a file whose default was restored by
    # hand: it looks for the marker, not for one particular value, which is why it survives
    # whatever the substitution produced
    import re
    bad = re.sub(r'SERIES_LIST=\$\{SERIES_LIST:-"[^"]*"\}',
                 'SERIES_LIST=${SERIES_LIST:-"@@SERIES@@"}', job)
    (tmp_path / "job2.sh").write_text(bad)
    r2 = subprocess.run(["bash", str(tmp_path / "job2.sh")], capture_output=True, text=True)
    assert r2.returncode == 6 and "marker" in r2.stdout, r2


def test_a_missing_endpoint_is_refused_before_dispatch(tmp_path):
    tpl = tmp_path / "tpl.sh"
    tpl.write_text(TEMPLATE)
    ser = tmp_path / "series.txt"
    ser.write_text("s1\n")
    _bridge(tmp_path, BRIDGE="/b", TRUX_SSH="ssh", SRC="/d")   # TRUX absent
    r = _run(tmp_path, tpl, ser)
    assert r.returncode == 6 and "TRUX" in r.stderr, r.stderr


def test_a_token_the_generator_does_not_know_fails_the_run(tmp_path):
    """An unknown placeholder must not be shipped as a literal."""
    tpl = tmp_path / "tpl.sh"
    tpl.write_text(TEMPLATE.replace('TRUX=${TRUX:-"@@TRUX@@"}',
                                    'OUT=${OUT:-"@@OUT_DIR@@/lane.jsonl"}'))
    ser = tmp_path / "series.txt"
    ser.write_text("s1\n")
    _bridge(tmp_path, BRIDGE="/b", TRUX="t", TRUX_SSH="ssh", SRC="/d")
    r = _run(tmp_path, tpl, ser)
    assert r.returncode != 0 and "OUT_DIR" in r.stderr, r.stderr
