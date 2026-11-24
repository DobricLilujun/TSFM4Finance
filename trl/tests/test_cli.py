"""CLI smoke tests on the statistical baselines: no weights, no GPU, no downloads.

These exist because the CLI is the surface a benchmark run actually touches, and a
broken argument is invisible to the API tests. `evaluate` in particular had a
`_load_model(a.model)` call that passed a string where a namespace was expected — every
`tsfm-trl evaluate` invocation died with ``'str' object has no attribute 'model'`` and
nothing in the suite noticed.
"""
from __future__ import annotations

import json

import pytest

from tsfm_trl.cli import main


def test_evaluate_runs_one_row_per_horizon(capsys):
    rc = main(["evaluate", "--model", "naive", "--horizons", "16,32",
               "--ctx", "96", "--max-windows", "8"])
    assert rc == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["horizon"] for r in rows] == [16, 32]
    for r in rows:
        assert r["span"] == "test" and r["n_windows"] >= 1
        # point metrics always; CRPS only if the model emits quantiles (naive does not)
        assert {"rmse", "mae", "mase"} <= set(r)


def test_evaluate_falls_back_to_single_horizon(capsys):
    assert main(["evaluate", "--model", "theta", "--horizon", "16", "--ctx", "96"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["horizon"] for r in rows] == [16]


def test_list_and_info_are_parseable(capsys):
    assert main(["list", "--tier", "A"]) == 0
    assert "chronos-2" in capsys.readouterr().out
    assert main(["info", "chronos-t5-small"]) == 0
    body = capsys.readouterr().out
    assert "amazon/chronos-t5-small" in body and "paths" in body


def test_bad_budget_is_rejected_before_any_weight_is_read(capsys):
    # argparse owns this one: a rejected choice must not reach the loader, let alone a
    # 4 GB download
    with pytest.raises(SystemExit):
        main(["fit", "--model", "naive", "--budget", "fs9999"])
    assert "invalid choice" in capsys.readouterr().err
