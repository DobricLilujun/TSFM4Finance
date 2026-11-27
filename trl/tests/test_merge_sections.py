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


def _meta(machine, code_sha="deadbeef", torch="2.7.1+cu128", device="cuda"):
    # real lane files nest the block: {"meta": {...}}, and the reader takes r["meta"]
    # keys mirror a real lane file's meta block; the report header reads them directly.
    # The stack fields are per-call because the stack-vs-protocol split hinges on them.
    return {"meta": {"started": "2026-01-01 00:00:00", "machine": machine,
                     "code_sha": code_sha, "python": "3.11.0", "torch": torch,
                     "device": device, "ctx": 256, "horizons": [21, 63], "spans": ["test"],
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
    assert "Superseded by a protocol change (what was measured differs): ttm-r2 (1)" in text, text
    assert "**0** pairs share a full signature" in text, text
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
    assert "Superseded by a protocol change (what was measured differs): chronos-bolt-small (1)" in text, text
    assert "**0** pairs share a full signature" in text, text


def _row(model, series, mase, rmse=1.0, ctx=256):
    return {"model": model, "family": "fam", "env": "core", "interpreter": "/venv/bin/python",
            "torch": None, "cuda": None, "dataset": "datasets/standard/x", "task": "forecast",
            "status": "ok", "detail": "ran", "error": None, "series_errors": {},
            "rows": [{"series": series, "span": "test", "horizon": 21, "ctx": ctx,
                      "mase": mase, "rmse": rmse, "mda": 0.5, "nd": 0.0, "no_leakage": True,
                      "n_windows": 2, "seconds": 1.0, "adaptation": "zero-shot",
                      "model": model}]}


def test_headline_ranking_compares_against_the_naive_row_not_mase_below_one(tmp_path):
    """The trap this guards: MASE's denominator is the *in-sample* naive error, so MASE < 1
    is not "beat naive on this cell". On these financial series the naive row itself scores
    ~4.5, and the first version of this table printed 0% for every model. The comparison is
    against the naive row scored on the same (series, span, horizon) cell."""
    a = tmp_path / "lane.jsonl"
    recs = [json.dumps(_meta("trux"))]
    # naive baseline per cell: 1.4 on s1/s2/s3, but a hard cell s4 where naive is 0.5
    for srs, nm in (("s1", 1.4), ("s2", 1.4), ("s3", 1.4), ("s4", 0.5)):
        recs.append(json.dumps(_row("naive", srs, nm)))
    recs += [
        json.dumps(_row("chronos-bolt-small", "s1", 0.5)),    # beats 1.4
        json.dumps(_row("chronos-bolt-small", "s2", 0.8)),    # beats 1.4
        json.dumps(_row("chronos-bolt-small", "s3", None, rmse=None)),  # no measurement
        json.dumps(_row("chronos-bolt-small", "s4", 0.9)),    # MASE < 1 but LOSES to 0.5
        json.dumps(_row("moment-1-large", "s1", 3.0)),
        json.dumps(_row("moment-1-large", "s2", 2.0)),
    ]
    a.write_text("\n".join(recs) + "\n")
    md = tmp_path / "M.md"
    _report((a,), md)
    text = md.read_text()
    h = text.index("## Headline ranking")
    rank = text[h:text.index("## ", h + 5)]
    cb = next(l for l in rank.splitlines() if "chronos-bolt-small" in l)
    mo = next(l for l in rank.splitlines() if "moment-1-large" in l)
    assert rank.index("chronos-bolt-small") < rank.index("moment-1-large"), "better model sorts first"
    # 4 cells, 3 comparable (s3 had no measurement), 2 wins: s4 is comparable and LOST,
    # even though its MASE 0.9 is below 1.0
    assert "67% (2/3)" in cb, f"s4 must not count as a win despite MASE 0.9 < 1: {cb}"
    assert "0/2" in mo, mo
    assert cb.rstrip().endswith("| 1 |"), f"the None-mase cell counts as no measurement: {cb}"
    assert "not MASE < 1" in rank, "the table must say which comparison it is making"
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


def test_the_window_sample_is_part_of_the_protocol_signature():
    """`--max-windows` subsamples the test span at random (same seed, same windows), so two
    rows of the same cell with different n_windows are different measurements. Without this
    in the signature, a 32-window re-run silently replaces an 8-window row and the merge
    report calls the difference run variance."""
    mod = _load()
    base = {"ctx": 256, "ctx_note": "", "data_kind": "arena_dir", "data_hash": "abc123",
            "n_windows": 8, "_prov": {"device": "cuda", "code_sha": "aaaaaaa",
                                      "torch": "2.7.1", "seed": 0}}
    r1 = dict(base)
    r2 = dict(base, n_windows=32)
    r3 = dict(base, _prov=dict(base["_prov"], seed=1))
    assert mod.proto(r1) != mod.proto(r2), "8 windows vs 32 windows is a different protocol"
    assert mod.proto(r1) != mod.proto(r3), "a different seed is a different window sample"
    assert mod.proto(r1) == mod.proto(dict(base)), "same protocol must still compare equal"


def test_the_ranking_publishes_the_window_sample(tmp_path):
    """`--max-windows` subsamples the test span uniformly at random, so the window count is part
    of what a row measured. A ranking that hides it hides the sampling design: at 8 windows the
    paired model/naive ratio moves ~12% median between seeds, at 32 windows ~7%."""
    lane = tmp_path / "lane.jsonl"
    lane.write_text(json.dumps(_meta("m1")) + "\n"
                    + json.dumps(_rec("chronos-bolt-small", "s1", "ok")) + "\n"
                    + json.dumps(_rec("naive", "s1", "ok")) + "\n")
    md = tmp_path / "r.md"
    _report([lane], md, summary_only=True)
    out = md.read_text()
    assert "| win |" in out, out[-500:]
    assert "max-windows" in out, "the sampling caveat must sit next to the table"


def test_the_same_bytes_on_another_machine_is_not_called_a_fix(tmp_path):
    """Pass 4/5 produced 1,452 cross-machine re-measurements of identical cells. Filing them
    under "superseded by a fix" would tell the reader the earlier numbers were wrong; they are
    a reproducibility measurement, so the report has to name which case it is showing."""
    def row(mase):
        return {"model": "chronos-bolt-small", "family": "fam", "env": "core",
                "interpreter": "/venv/bin/python", "torch": None, "cuda": None,
                "dataset": "d", "task": "forecast", "status": "ok", "detail": "",
                "error": None, "series_errors": {},
                "rows": [{"series": "s1", "span": "all", "horizon": 21, "ctx": 512,
                          "mase": mase, "rmse": 5.0, "mda": 0.4, "nd": 0.0,
                          "no_leakage": True, "ctx_note": "", "n_windows": 8,
                          "data_kind": "arena_dir", "data_hash": "abc123abc123"}]}

    a = tmp_path / "gpu7.jsonl"
    b = tmp_path / "watson.jsonl"
    a.write_text("\n".join([json.dumps(_meta("gpu7", code_sha="aaaaaaa",
                                             torch="2.9.1+cu130", device="cuda")),
                            json.dumps(row(4.2))]) + "\n")
    b.write_text("\n".join([json.dumps(_meta("sedan-gpu3", code_sha="bbbbbbb",
                                             torch="2.7.1+cu128", device="cuda")),
                            json.dumps(row(4.2))]) + "\n")
    md = tmp_path / "out.md"
    _report([a, b], md)
    text = md.read_text()
    assert "Re-measured on another stack (same bytes, same protocol): chronos-bolt-small (1)" in text, text
    assert "Superseded by a protocol change" not in text, text


def test_a_different_window_sample_is_a_protocol_change_not_a_stack(tmp_path):
    """Eight windows vs 32 windows on one machine is a different sample of the same series.
    The gap between those rows is the sampling artifact (~12% median on the paired ratio), and
    it must not be filed under reproducibility."""
    def row(mase, nw):
        return {"model": "naive", "family": "fam", "env": "core",
                "interpreter": "/venv/bin/python", "torch": None, "cuda": None,
                "dataset": "d", "task": "forecast", "status": "ok", "detail": "",
                "error": None, "series_errors": {},
                "rows": [{"series": "s1", "span": "all", "horizon": 21, "ctx": 512,
                          "mase": mase, "rmse": 5.0, "mda": 0.4, "nd": 0.0,
                          "no_leakage": True, "ctx_note": "", "n_windows": nw,
                          "data_kind": "arena_dir", "data_hash": "abc123abc123"}]}

    a = tmp_path / "w8.jsonl"
    b = tmp_path / "w32.jsonl"
    a.write_text("\n".join([json.dumps(_meta("gpu7")),
                            json.dumps(row(4.0, 8))]) + "\n")
    b.write_text("\n".join([json.dumps(_meta("gpu7")),
                            json.dumps(row(5.2, 32))]) + "\n")
    md = tmp_path / "out.md"
    _report([a, b], md)
    text = md.read_text()
    assert "Superseded by a protocol change (what was measured differs): naive (1)" in text, text
    assert "Re-measured on another stack" not in text, text
