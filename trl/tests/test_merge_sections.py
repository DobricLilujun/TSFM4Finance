"""Report sections must count merged cells, not one raw record per model.

`rec_by_model` keeps a single raw record per model (the one with more rows). The tables are
built from the merged `cells`, but the two summary sections -- dataset coverage gaps and the
untrained-head flag -- used to iterate `recs`. With two GPU lanes each covering half the
corpus, or a GPU lane plus a CPU retry, that silently reported one lane: pass 3 showed 33
of 66 flagged MOMENT series, and pass 2 under-counted the covariate-panel gaps.
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


def _report(files, md_path):
    """Call the generator the way the CLI does (it takes an argparse namespace, not paths):
    a subprocess run would hide the traceback, which is how this test started."""
    _load().report(argparse.Namespace(report=",".join(str(f) for f in files),
                                      md=str(md_path), stdout=False))


def _meta(machine):
    # real lane files nest the block: {"meta": {...}}, and the reader takes r["meta"]
    # keys mirror a real lane file's meta block; the report header reads them directly
    return {"meta": {"started": "2026-01-01 00:00:00", "machine": machine,
                     "code_sha": "deadbeef", "python": "3.11.0", "torch": "2.7.1+cu128",
                     "device": "cuda", "ctx": 256, "horizons": [21, 63], "spans": ["test"],
                     "max_windows": 8, "batch": 16, "seed": 0}}


def _rec(model, series, flag, dataset="datasets/standard/x"):
    row = {"series": series, "span": "test", "horizon": 21, "ctx": 256, "mase": 1.4, "mda": 0.5,
           "nd": 0.0, "coverage_40": None, "nominal_40": None, "wql": None, "smape": None,
           "point_mase": None, "metric_mase": None, "no_leakage": True, "n_windows": 2,
           "seconds": 1.0, "adaptation": "zero-shot", "model": model,
           "meta": {"head_from_checkpoint": not flag, "head_trained_here": False}}
    return {"model": model, "family": "fam", "env": "core", "interpreter": "/venv/bin/python",
            "torch": None, "cuda": None, "dataset": dataset, "task": "forecast",
            "status": "ok", "detail": "ran", "error": None, "series_errors": {},
            "rows": [row]}


def test_sections_cover_every_lane(tmp_path):
    a = tmp_path / "lane-a.jsonl"
    b = tmp_path / "lane-b.jsonl"
    a.write_text("\n".join([json.dumps(_meta("trux")),
                            json.dumps(_rec("moment-1-small", "s1", True)),
                            json.dumps(_rec("kronos", "s1", False))]) + "\n")
    b.write_text("\n".join([json.dumps(_meta("trux")),
                            json.dumps(_rec("moment-1-small", "s2", True)),
                            json.dumps(_rec("kronos", "s2", False))]) + "\n")
    md = tmp_path / "M.md"
    _report((a, b), md)
    text = md.read_text()
    flag = text[text.index("## Flagged"):]
    assert "**moment-1-small**: 2 series" in flag, "flagged set must span every lane, not one record"
    assert "kronos" not in flag.splitlines()[-1] and "kronos**" not in flag
    assert "s1" in flag and "s2" in flag


def test_a_model_scored_somewhere_is_not_reported_unscoreable(tmp_path):
    a = tmp_path / "lane-a.jsonl"
    b = tmp_path / "lane-b.jsonl"
    err = _rec("kronos", "s1", False)
    err["rows"], err["status"], err["error"] = [], "failed", (
        "ValueError: kronos scores a covariate panel (open/high/low/close); this series is close-only")
    a.write_text("\n".join([json.dumps(_meta("trux")), json.dumps(err)]) + "\n")
    b.write_text("\n".join([json.dumps(_meta("trux")), json.dumps(_rec("kronos", "s2", False))]) + "\n")
    md = tmp_path / "M.md"
    _report((a, b), md)
    assert "Not scoreable on some series" not in md.read_text(), (
        "a model that scored on any lane is not dataset-unscoreable")
