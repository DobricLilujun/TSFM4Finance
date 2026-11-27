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


def _report(files, md_path, summary_only=False):
    """Call the generator the way the CLI does (it takes an argparse namespace, not paths):
    a subprocess run would hide the traceback, which is how this test started."""
    _load().report(argparse.Namespace(report=",".join(str(f) for f in files),
                                      md=str(md_path), stdout=False,
                                      summary_only=summary_only))


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


def test_duplicate_cell_keys_are_reported_not_silently_resolved(tmp_path, capsys):
    """The same protocol measured twice in one file is run variance, not a retry.

    (This test first shipped when the two chronos-bolt rows in lane_cuda-b.jsonl - MASE
    2.2547 and 2.3330 - looked like the same cell measured twice. They were not: the two
    rows came from chronos-bolt-small and chronos-bolt-base sharing a row-level model id
    (see test_model_key_collision.py). Same-checkpoint duplicates do exist, mostly in the
    TTM retries; the merge must still report them.
    """
    def row(series, horizon, mase):
        return {"model": "chronos-bolt-small", "adaptation": "zero", "series": series,
                "span": "all", "horizon": horizon, "ctx": 256, "n_windows": 8,
                "mase": mase, "no_leakage": True}

    lane = tmp_path / "lane.jsonl"
    rec = {"model": "chronos-bolt-small", "env": "core", "device": "cuda", "interpreter":
           "/x/core/bin/python", "torch": {"torch": "2.7.1+cu128"},
           "rows": [row("s1", 21, 2.2547), row("s1", 21, 2.3330), row("s2", 21, 1.5)]}
    lane.write_text("\n".join([json.dumps(_meta("trux")), json.dumps(rec)]) + "\n")

    md = tmp_path / "out.md"
    _report([lane], md)
    text = md.read_text()
    assert "## Repeated cell keys: 1" in text, text
    assert "run variance" in text
    assert "max 3.36%" in text, text           # |2.2547-2.3330| / 2.3330
    # newest measurement still wins the cell (the table renders MASE to 2dp)
    table = text.split("## Repeated cell keys")[0]
    assert "2.33" in table and "2.25" not in table
    err = capsys.readouterr().err
    assert "appear more than once" in err


def test_a_repeated_key_under_a_changed_protocol_is_not_run_variance(tmp_path, capsys):
    """A cell scored before and after a fix is a supersession, and the gap is the fix.

    The pass-3 merge's 404 repeated keys are mostly this: TTM at ctx 256 (the zero-padding
    artifact, MASE ~30) then the same cell at its contract ctx 512 (MASE ~4). Reporting
    that as "run variance up to 10%" would tell readers the table is noisy when the noisy
    half is the bug we removed. The protocol signature - ctx, ctx_note, device, code_sha,
    torch - decides which reading applies.
    """
    def row(ctx, mase):
        return {"model": "ttm-r2", "adaptation": "zero", "series": "s1", "span": "all",
                "horizon": 21, "ctx": ctx, "n_windows": 8, "mase": mase, "no_leakage": True,
                "ctx_note": "" if ctx == 256
                else "context raised 256->512 (checkpoint contract)"}

    def rec(ctx, mase):
        return {"model": "ttm-r2", "env": "core", "device": "cuda",
                "interpreter": "/x/core/bin/python", "torch": {"torch": "2.7.1+cu128"},
                "rows": [row(ctx, mase)]}

    a = tmp_path / "old.jsonl"
    b = tmp_path / "new.jsonl"
    a.write_text("\n".join([json.dumps(_meta("trux")), json.dumps(rec(256, 30.1047))]) + "\n")
    b.write_text("\n".join([json.dumps(_meta("trux")), json.dumps(rec(512, 4.3169))]) + "\n")

    md = tmp_path / "out.md"
    _report([a, b], md)
    text = md.read_text()
    assert "Superseded by a protocol change: ttm-r2 (1)" in text, text
    assert "**0** share a signature" in text, text
    assert "appear more than once" in capsys.readouterr().err


def test_a_cell_read_from_another_data_path_is_a_supersession_not_run_variance(tmp_path, capsys):
    """Same model, same series name, same ctx: the CSV lane and the arena lane are not the
    same measurement.

    The cross-machine check found watson's cells 15% off trux's on MASE while RMSE matched
    to eight decimals. Both lanes read "the same series", but one loaded a flat CSV (whose
    train split is the synthetic 70%) and the other an arena directory (whose train split is
    the publisher's), and MASE is divided by the naive error of whichever one was loaded. The
    fingerprint on the row is what lets the merge file that under "superseded" instead of
    "noise".
    """
    def row(kind, h, mase):
        return {"model": "chronos-bolt-small", "adaptation": "zero", "series": "s1",
                "span": "all", "horizon": 21, "ctx": 256, "n_windows": 8, "mase": mase,
                "no_leakage": True, "ctx_note": "",
                "data_kind": kind, "data_hash": h}

    def rec(kind, h, mase):
        return {"model": "chronos-bolt-small", "env": "core", "device": "cuda",
                "interpreter": "/x/core/bin/python", "torch": {"torch": "2.7.1+cu128"},
                "rows": [row(kind, h, mase)]}

    a = tmp_path / "csv_lane.jsonl"
    b = tmp_path / "arena_lane.jsonl"
    a.write_text("\n".join([json.dumps(_meta("watson")),
                            json.dumps(rec("csv", "aaaa11112222", 5.3916))]) + "\n")
    b.write_text("\n".join([json.dumps(_meta("gpu7")),
                            json.dumps(rec("arena_dir", "bbbb33334444", 6.1991))]) + "\n")

    md = tmp_path / "out.md"
    _report([a, b], md)
    text = md.read_text()
    assert "Superseded by a protocol change: chronos-bolt-small (1)" in text, text
    assert "**0** share a signature" in text, text


def _row(model, series, mase, rmse=1.0, ctx=256):
    return {"model": model, "family": "fam", "env": "core", "interpreter": "/venv/bin/python",
            "torch": None, "cuda": None, "dataset": "datasets/standard/x", "task": "forecast",
            "status": "ok", "detail": "ran", "error": None, "series_errors": {},
            "rows": [{"series": series, "span": "test", "horizon": 21, "ctx": ctx,
                      "mase": mase, "rmse": rmse, "mda": 0.5, "nd": 0.0, "no_leakage": True,
                      "n_windows": 2, "seconds": 1.0, "adaptation": "zero-shot",
                      "model": model}]}


def test_headline_ranking_counts_beat_naive_and_skips_missing_measurements(tmp_path):
    """The aggregate table is the site's headline: beat-naive share per model, with
    no-measurement cells counted rather than averaged into the median."""
    a = tmp_path / "lane.jsonl"
    a.write_text("\n".join([
        json.dumps(_meta("trux")),
        json.dumps(_row("chronos-bolt-small", "s1", 0.5)),
        json.dumps(_row("chronos-bolt-small", "s2", 0.8)),
        json.dumps(_row("chronos-bolt-small", "s3", None, rmse=None)),
        json.dumps(_row("moment-1-large", "s1", 3.0)),
        json.dumps(_row("moment-1-large", "s2", 2.0)),
        json.dumps(_row("moment-1-large", "s3", 4.0)),
    ]) + "\n")
    md = tmp_path / "M.md"
    _report((a,), md)
    text = md.read_text()
    rank = text[text.index("## Headline ranking"):text.index("## ", text.index("## Headline ranking") + 5)]
    cb = next(l for l in rank.splitlines() if "chronos-bolt-small" in l)
    mo = next(l for l in rank.splitlines() if "moment-1-large" in l)
    assert rank.index("chronos-bolt-small") < rank.index("moment-1-large"), \
        "the better model must sort first"
    assert "| 3 | 67% |" in cb, f"2 of 3 cells beat naive, third is not a measurement: {cb}"
    assert "| 3 | 0% |" in mo, mo
    assert cb.rstrip().endswith("| 1 |"), f"the None-mase cell is counted as no measurement: {cb}"
    assert "nan" not in rank, "a missing measurement must never render as nan"


def test_summary_only_keeps_aggregates_and_drops_per_cell_tables(tmp_path):
    """GitHub Pages does not serve the directory URL of a ~2 MB page, and a 3,600-row
    table is unreadable anyway: the site asks for aggregates only."""
    a = tmp_path / "lane.jsonl"
    a.write_text("\n".join([json.dumps(_meta("trux")),
                            json.dumps(_row("chronos-bolt-small", "s1", 0.5)),
                            json.dumps(_row("chronos-bolt-small", "s2", 0.7))]) + "\n")
    md = tmp_path / "M.md"
    _report((a,), md, summary_only=True)
    text = md.read_text()
    assert "## Headline ranking" in text
    assert "## Which interpreter produced which row" in text
    assert "· horizon 21" not in text, "per-cell tables must be absent in summary mode"
    assert "--summary-only" in text, "the page must say what it left out and why"
