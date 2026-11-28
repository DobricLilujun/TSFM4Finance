"""Import-and-scope contract for the bench tree.

Three silent breaks lived in this code and none of them raised at import time:

* `bench/probe2.py::_cv_predict` used sklearn's `clone` without importing it. The call sat inside
  `try: ... except Exception: return None`, so the NameError was caught and every caller silently
  got `None` instead of out-of-fold probabilities - the anti-contamination rule quietly stopped
  applying to every classification probe.
* `bench/run.py` read `args.ft_epochs` inside `stage_finetune`/`stage_valcheck`, where `args` is
  not in scope (it is local to `main`). Any fine-tune stage would have died with NameError.
* `bench/download/__init__.py::prefetch` used `t4_scrape` without loading it through `_load`, the
  helper the package needs because a module-level name shadows the submodule of the same name.

The clone bug is pinned functionally; the two scope bugs are pinned by source text, in the same
spirit as the existing "fine-tune contract" test: these are one-line invariants that a refactor
keeps breaking.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parents[1] / "tsfm4finance" / "bench"


def test_cv_predict_actually_returns_out_of_fold_probabilities():
    """Before the import fix this returned None for every input."""
    from sklearn.linear_model import LogisticRegression

    from tsfm4finance.bench.probe2 import _cv_predict

    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(-1.5, 0.6, (60, 3)), rng.normal(1.5, 0.6, (60, 3))])
    y = np.array([0] * 60 + [1] * 60)
    proba = _cv_predict(LogisticRegression(max_iter=500), X, y, n_splits=3)
    assert proba is not None, "the classifier was never cloned: the NameError was swallowed"
    assert proba.shape == (120, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_run_passes_the_fine_tune_epoch_count_through_a_module_global():
    src = (BENCH / "run.py").read_text()
    # reading args inside main() is fine; reading it from a stage function is the bug
    assert "epochs=args.ft_epochs" not in src, "args is local to main(); the stage functions cannot see it"
    assert "FT_EPOCHS: int | None = None" in src
    assert "epochs=FT_EPOCHS," in src
    assert "global FT_LR, FT_CONTEXT, FT_BS, FT_EPOCHS, HORIZONS, CONTEXTS" in src


def test_the_downloader_loads_every_tier_module_it_calls():
    src = (BENCH / "download" / "__init__.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "prefetch")
    loaded = {a.value for n in ast.walk(fn)                      # _load("name")
              if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_load"
              for a in n.args if isinstance(a, ast.Constant)}
    used = {n.func.value.id for n in ast.walk(fn)                # something.run(...)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id not in ("out", "json", "manifest_read")}
    assert "t4_scrape" in loaded, "t4 tier referenced a module that was never imported"
    undeclared = {u for u in used if u not in loaded and u not in {"_load"}}
    assert not undeclared, f"tier modules called without _load: {sorted(undeclared)}"


def test_the_stooq_fallback_table_has_no_repeated_key():
    """A duplicated key in a dict literal is legal Python and the later value wins, so a ticker
    can silently change what it resolves to while the file still looks fine."""
    src = (BENCH / "download" / "yf_panel.py").read_text()
    tree = ast.parse(src)
    assign = next(n for n in tree.body if isinstance(n, ast.Assign)
                  and getattr(n.targets[0], "id", "") == "STOOQ_FALLBACK")
    keys = [k.value for k in assign.value.keys]                  # type: ignore[union-attr]
    dupes = {k for k in keys if keys.count(k) > 1}
    assert not dupes, f"STOOQ_FALLBACK lists {sorted(dupes)} twice"
    assert len(set(keys)) == len(keys)
