"""Regression: merging several sweep JSONLs must not delete real measurements.

Multi-machine sweeps are partial by nature — one machine scored a model on two series,
another on the third. Whole-record merging (the first version) kept only the newest
record per model and silently dropped the other machine's rows.
"""
import argparse
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "sweep_evaluate.py"


def _load():
    spec = importlib.util.spec_from_file_location("sweep_evaluate", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rec(model, series, mase, error=None):
    return {"model": model, "env": "core", "tier": "A", "family": "f",
            "interpreter": "/venv/bin/python", "device": "cpu",
            "torch": {"torch": "2.0", "cuda": False},
            "rows": [{"model": model, "series": series, "span": "all", "horizon": 21,
                      "ctx": 256, "n_windows": 3, "mase": mase, "rmse": mase, "mae": mase,
                      "dir_acc": 0.5, "rank_corr": 0.1, "relative_skill": -1.0}],
            "error": error or "", "series_errors": {}}


def test_partial_coverage_from_two_machines_is_unioned(tmp_path):
    mod = _load()
    f1 = tmp_path / "older.jsonl"
    f2 = tmp_path / "newer.jsonl"
    f1.write_text("\n".join(json.dumps(x) for x in [
        {"meta": {"machine": "a", "code_sha": "1", "series": ["s1"], "horizons": [21],
                 "spans": "all", "max_windows": 3, "ctx": 256, "batch": 16, "seed": 0,
                 "device": "cpu"}},
        _rec("modelx", "s1", 1.5)]))
    f2.write_text("\n".join(json.dumps(x) for x in [
        {"meta": {"machine": "b", "code_sha": "2", "series": ["s2"], "horizons": [21],
                 "spans": "all", "max_windows": 3, "ctx": 256, "batch": 16, "seed": 0,
                 "device": "cuda"}},
        {"model": "modelx", "env": "core", "tier": "A", "family": "f",
         "interpreter": "/other/venv/bin/python", "device": "cuda",
         "torch": {"torch": "2.1", "cuda": True}, "rows": [
             {"model": "modelx", "series": "s2", "span": "all", "horizon": 21,
              "ctx": 256, "n_windows": 3, "mase": 2.5, "rmse": 2.5, "mae": 2.5,
              "dir_acc": 0.4, "rank_corr": 0.2, "relative_skill": -1.5}],
         "error": "", "series_errors": {}}]))
    md = tmp_path / "out.md"
    args = argparse.Namespace(report=f"{f1},{f2}", md=str(md), stdout=False)
    assert mod.report(args) == 0
    text = md.read_text()
    # both series must appear with their own numbers, and each with its own provenance
    assert "1.5" in text and "2.5" in text, text
    assert text.count("modelx") >= 2
    assert "/venv/bin/python" in text and "/other/venv/bin/python" in text


def test_error_only_run_does_not_erase_a_row(tmp_path):
    mod = _load()
    f1 = tmp_path / "good.jsonl"
    f2 = tmp_path / "later_error.jsonl"
    f1.write_text(json.dumps(_rec("modelx", "s1", 1.5)))
    f2.write_text(json.dumps({"model": "modelx", "env": "core", "tier": "A", "family": "f",
                              "interpreter": "/v/bin/python", "device": "cpu", "rows": [],
                              "error": "boom", "series_errors": {}}))
    md = tmp_path / "out.md"
    args = argparse.Namespace(report=f"{f1},{f2}", md=str(md), stdout=False)
    assert mod.report(args) == 0
    assert "1.5" in md.read_text()
