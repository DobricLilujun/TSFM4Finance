"""The data path is part of the measurement, not of the model.

A pass-3 cross-machine check reported 15-123% MASE disagreement between two GPUs while
RMSE agreed to the eighth decimal. Neither side was wrong and no forecast differed: watson
had loaded its series from exported CSVs (synthetic 70/15/15 split) and the other lane from
arena directories (the publisher's train/validation/test split). MASE is divided by the
naive error of whichever train split was loaded, so the *denominator* differed by 1.15x on
that series, and every scaled metric moved with it.

These tests pin the three things that make that visible in the data instead of in a
post-mortem: a fingerprint per load, the fingerprint on every scored row, and a merge that
refuses to call a data-path change "run variance".
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent.parent

import json
import numpy as np
import pandas as pd
import pytest

from tsfm_trl.data.series import SeriesDataset
from tsfm_trl.eval.evaluate import forecast_eval
from tsfm_trl.model.base import Pred


class _NaiveStub:
    """Last value repeated: deterministic, so any metric difference between two loads of
    the same numbers is a data-path effect and nothing else."""

    key = "stub_naive"
    caps = ()
    max_context = 512
    ctx_step = 1
    adaptation = "zero"

    def predict(self, contexts, horizon, panels=None):
        pts = np.tile(np.asarray(contexts, dtype=float)[:, -1:], (1, horizon))
        return Pred(point=pts.reshape(len(contexts), horizon), meta={})


def _values(n=900):
    rng = np.random.default_rng(7)
    return np.cumsum(rng.normal(0.0, 1.0, n)) + 100.0


@pytest.fixture()
def two_paths(tmp_path):
    """One series, two legal input shapes, identical target values."""
    v = _values()
    idx = pd.date_range("2000-01-01", periods=len(v), freq="D")
    d = tmp_path / "arena_series"
    d.mkdir()
    tr, va = int(0.55 * len(v)), int(0.15 * len(v))
    parts = {"train": v[:tr], "validation": v[tr:tr + va], "test": v[tr + va:]}
    for name, arr in parts.items():
        start = {"train": 0, "validation": tr, "test": tr + va}[name]
        pd.DataFrame({"timestamp": idx[start:start + len(arr)],
                      "close": arr}).to_parquet(d / f"{name}.parquet")
    (d / "meta.json").write_text(json.dumps({"name": "arena_series", "target": "close"}))

    csv = tmp_path / "arena_series.csv"
    pd.DataFrame({"timestamp": idx, "close": v}).to_csv(csv, index=False)
    return d, csv, v


def test_the_same_bytes_load_to_the_same_fingerprint(two_paths):
    d, csv, v = two_paths
    a, b = SeriesDataset.load(d), SeriesDataset.load(d)
    assert a.meta["data_hash"] == b.meta["data_hash"]
    assert a.meta["data_kind"] == "arena_dir"
    assert SeriesDataset.load(csv).meta["data_hash"] == SeriesDataset.load(csv).meta["data_hash"]


def test_a_csv_and_the_arena_directory_of_one_series_are_different_measurements(two_paths):
    d, csv, v = two_paths
    arena, flat = SeriesDataset.load(d), SeriesDataset.load(csv)
    assert np.allclose(arena.values, flat.values), "same series, same values"
    assert arena.meta["data_hash"] != flat.meta["data_hash"], (
        "the fingerprint must separate a flat file from the arena directory: the split "
        "boundaries differ, and with them every scaled metric")
    assert arena.meta["data_kind"] == "arena_dir"
    assert flat.meta["data_kind"] == "csv"
    # and the split really is different: arena keeps the publisher's 55% train boundary,
    # the flat file gets the synthetic 70%
    assert arena.splits["train"][1] != flat.splits["train"][1]


def test_mase_is_the_naive_error_of_whichever_split_was_loaded(two_paths):
    d, csv, v = two_paths
    arena, flat = SeriesDataset.load(d), SeriesDataset.load(csv)
    m = _NaiveStub()
    ra = forecast_eval(m, arena, horizons=(21,), spans=("test",), ctx=120)[0]
    rf = forecast_eval(m, flat, horizons=(21,), spans=("test",), ctx=120)[0]
    for tag, ds, row in (("arena", arena, ra), ("csv", flat, rf)):
        den = float(np.mean(np.abs(np.diff(ds.split("train")))))
        assert row["mase"] == pytest.approx(row["mae"] / den, rel=1e-9), (tag, row["mase"], den)
        assert row["data_hash"] == ds.meta["data_hash"]
    assert ra["data_kind"] == "arena_dir" and rf["data_kind"] == "csv"
    # the two paths are not even scoring the same windows: the arena test span is 30% of
    # the series, the flat file's is 15%, so n_windows is 7 vs 1 here. That is the deeper
    # half of the finding - a data path changes which cells exist, not only how they scale.
    assert ra["n_windows"] != rf["n_windows"], (ra["n_windows"], rf["n_windows"])


def test_the_sweep_stamps_the_fingerprint_on_every_row_and_on_the_lane(tmp_path):
    """A measurement whose data path nobody recorded cannot be reconciled later. The pass-3
    lanes are exactly that: watson read CSVs, trux read arena directories, and nothing in
    the files said so. Both levels are stamped now - the lane's meta line and each row."""
    import json
    import subprocess
    import sys

    d = tmp_path / "arena"
    d.mkdir()
    v = _values(400)
    idx = pd.date_range("2000-01-01", periods=400, freq="D")
    for nm, st, sp in (("train", 0, 220), ("validation", 220, 60), ("test", 280, 120)):
        pd.DataFrame({"timestamp": idx[st:st + sp], "close": v[st:st + sp]}
                     ).to_parquet(d / f"{nm}.parquet")
    (d / "meta.json").write_text(json.dumps({"name": "arena", "target": "close"}))
    out = tmp_path / "lane.jsonl"

    # the sweep dispatches families through the env resolver (cwd envs/ or envs/local.json),
    # not through the interpreter that launched it: without a venv to point at, every model
    # comes back as an env-skip record with zero rows
    (tmp_path / "envs").mkdir()
    (tmp_path / "envs" / "local.json").write_text(json.dumps({"core": sys.executable}))

    subprocess.run([sys.executable,
                    str(HERE / "scripts" / "sweep_evaluate.py"),
                    "--series", str(d), "--models", "naive", "--horizons", "10",
                    "--max-windows", "2", "--ctx", "32", "--seed", "0",
                    "--envs", str(tmp_path / "envs"),
                    "--out", str(out), "--device", "cpu", "--fresh"],
                   check=True, capture_output=True, text=True, timeout=300)

    recs = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    meta = next(r["meta"] for r in recs if "meta" in r)
    assert meta["data_kinds"] == ["arena_dir"], meta
    rows = [row for r in recs if r.get("rows") for row in r["rows"]]
    assert rows, recs
    for row in rows:
        assert row["data_kind"] == "arena_dir", row
        assert len(row["data_hash"]) == 12, row
