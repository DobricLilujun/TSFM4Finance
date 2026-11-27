"""Regression: the sweep's path normalisation must not leave the venv.

`Path.resolve()` follows `.venv/bin/python`'s symlink to the interpreter that built the
venv, which has none of the family's pinned packages; the run then fails on
`import numpy` and the error reads like a model/environment fault. See
docs/08_known_issues.md.
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "sweep_evaluate.py"


def _load():
    spec = importlib.util.spec_from_file_location("sweep_evaluate", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_abspath_keeps_a_symlinked_venv_python_in_its_venv(tmp_path):
    mod = _load()
    base = tmp_path / "base"
    venv_bin = tmp_path / ".venv-x" / "bin"
    base.mkdir()
    venv_bin.mkdir(parents=True)
    link = venv_bin / "python"
    link.symlink_to(base / "python3")           # what uv/venv actually do
    assert os.path.realpath(link) != str(link)  # the trap exists
    got = mod._abspath(str(link))
    assert got == str(link), got
    assert ".venv-x" in got


@pytest.mark.skipif(not Path(sys.executable).exists(), reason="interpreter moved")
def test_abspath_of_a_relative_path_stays_absolute():
    mod = _load()
    got = mod._abspath("envs/core/bin/python")
    assert os.path.isabs(got) and got.endswith("envs/core/bin/python")
