"""The docs build must pick an interpreter that can actually import mkdocs.

CI died with `/bin/python: No module named mkdocs`: with VIRTUAL_ENV unset, the old
fallback expression evaluated to the literal string `/bin/python` (empty string plus
suffix), which exists on GitHub runners and has no mkdocs, so it beat `python3` on PATH -
the one interpreter the workflow had pip-installed into.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "build_docs.sh"


def test_the_interpreter_falls_back_to_path_python3(tmp_path):
    """With no DOCS_PY and no VIRTUAL_ENV, the script must use `python3` from PATH."""
    marker = tmp_path / "picked.txt"
    shim = tmp_path / "python3"
    shim.write_text(f"#!/bin/sh\necho \"$0\" > {marker}\nexit 0\n")
    shim.chmod(0o755)

    env = dict(os.environ)
    # on a GitHub runner there is no .venv-docs in the checkout, and no VIRTUAL_ENV:
    # pin DOCS_PY to something that does not exist so the fallback path is what runs
    env["DOCS_PY"] = str(tmp_path / "no-such-interpreter")
    env.pop("VIRTUAL_ENV", None)
    env["PATH"] = f"{tmp_path}{os.pathsep}{env['PATH']}"

    r = subprocess.run(["bash", str(SCRIPT)], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-400:]
    assert marker.exists(), "the build never invoked python3 from PATH"
    assert str(shim) in marker.read_text(), marker.read_text()


def test_a_set_virtualenv_wins_over_path(tmp_path):
    """A real virtualenv is the documented override and must be preferred."""
    marker = tmp_path / "picked2.txt"
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    py = venv / "python"
    py.write_text(f"#!/bin/sh\necho \"$0\" > {marker}\nexit 0\n")
    py.chmod(0o755)
    shim = tmp_path / "python3"
    shim.write_text(f"#!/bin/sh\necho \"SHIM $0\" > {marker}\nexit 0\n")
    shim.chmod(0o755)

    env = dict(os.environ)
    env["DOCS_PY"] = str(tmp_path / "no-such-interpreter")
    env["VIRTUAL_ENV"] = str(tmp_path / "venv")
    env["PATH"] = f"{tmp_path}{os.pathsep}{env['PATH']}"

    r = subprocess.run(["bash", str(SCRIPT)], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-400:]
    got = marker.read_text()
    assert "SHIM" not in got and str(py) in got, got
