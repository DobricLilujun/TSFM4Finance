"""The cross-machine checker must refuse comparisons that are not comparisons.

Two ways a pair of rows is not evidence about hardware:

* different bytes under the same series name (CSV split vs arena directory): the MASE
  denominator changes, the forecast does not;
* different window sample: `--max-windows` subsamples the test span uniformly at random, so
  seed 0 and seed 1 (or 8 windows vs 32) measure different subsets of the same series. Naive's
  own MASE moves ~51% median between seeds at 8 windows - blaming that on a GPU is how the
  "neural cells move 2-200% run to run" claim got published and retracted.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SCRIPT = HERE / "probes" / "trux" / "cross_machine_check.py"


def _lane(path, machine, series, nwin, mase, data_hash, kind="arena_dir"):
    rows = [{"model_key": "chronos-bolt-small", "series": series, "span": "test",
             "horizon": 21, "mase": mase, "rmse": 1.0, "n_windows": nwin,
             "data_hash": data_hash, "data_kind": kind, "ctx": 256}]
    rec = {"meta": {"machine": machine, "code_sha": "aaaaaaa", "device": "cpu",
                    "series": [f"/data/{series}"], "data_kinds": [kind], "seed": 0}}
    body = {"model": "chronos-bolt-small", "status": "ok", "error": None,
            "interpreter": "/venv/bin/python", "device": "cpu", "torch": "2.7.1",
            "rows": rows}
    Path(path).write_text(json.dumps(rec) + "\n" + json.dumps(body) + "\n")


def _run(tmp_path, a, b):
    return subprocess.run([sys.executable, str(SCRIPT), str(a), str(b), "--tolerance", "0.02"],
                          capture_output=True, text=True, timeout=120)


def test_a_different_window_sample_is_not_reported_as_a_machine_difference(tmp_path):
    a, b = tmp_path / "lane-a.jsonl", tmp_path / "lane-b.jsonl"
    _lane(a, "trux", "s1", 8, 4.0, "abc123abc123")
    _lane(b, "gpu7", "s1", 32, 6.4, "abc123abc123")
    r = _run(tmp_path, a, b)
    out = r.stdout
    assert "different window sample" in out, out[-400:]
    assert "outside tolerance" not in out, "a 60% gap from 8 vs 32 windows is not a hardware fault"
    # machine labels are sorted, so which side prints first is not ours to assume
    assert "n_windows 8 != 32" in out or "n_windows 32 != 8" in out, out[-400:]


def test_same_window_sample_still_comparisons_normally(tmp_path):
    a, b = tmp_path / "lane-a.jsonl", tmp_path / "lane-b.jsonl"
    _lane(a, "trux", "s1", 8, 4.0, "abc123abc123")
    _lane(b, "gpu7", "s1", 8, 4.0, "abc123abc123")
    r = _run(tmp_path, a, b)
    assert "different window sample" not in r.stdout
    assert "No cell outside tolerance" in r.stdout or "outside tolerance" not in r.stdout, r.stdout[-400:]


def test_different_data_path_is_still_a_gap(tmp_path):
    a, b = tmp_path / "lane-a.jsonl", tmp_path / "lane-b.jsonl"
    _lane(a, "trux", "s1", 8, 4.0, "abc123abc123", kind="arena_dir")
    _lane(b, "gpu7", "s1", 8, 4.6, "def456def456", kind="csv")
    r = _run(tmp_path, a, b)
    assert "could not be compared" in r.stdout, r.stdout[-400:]
    assert "outside tolerance" not in r.stdout
