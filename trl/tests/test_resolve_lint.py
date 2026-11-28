"""`scripts/check.sh` must stop when there is no linter, and use one from anywhere on the machine.

The gate that was supposed to catch lint stopped doing it on trux: the harness env has no ruff, so
`$PY -m ruff` died at the first line of the gate and everything after it - tests, registry, matrix -
never ran. The fix is a resolver with three accepted sources and an explicit failure, pinned here.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SCRIPT = HERE / "scripts" / "resolve_lint.sh"


def _run(path: Path, env: dict, py: str = ""):
    return subprocess.run(["bash", str(SCRIPT), py], capture_output=True, text=True,
                          env={**os.environ, **env}, timeout=300)


def test_the_linter_is_resolved_in_a_declared_order(tmp_path):
    # a ruff on PATH wins over uvx, and the resolver prints the command, nothing else
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "ruff").write_text('#!/bin/sh\necho "ruff 0.14.0"\n')
    (fake / "ruff").chmod(0o755)
    p = _run(SCRIPT, {"PATH": f"{fake}:/usr/bin:/bin"})
    assert p.returncode == 0, p.stderr[-300:]
    assert p.stdout.strip() == "ruff", p.stdout


def test_a_machine_with_only_uvx_still_lints(tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "uvx").write_text('#!/bin/sh\necho uvx\n')
    (fake / "uvx").chmod(0o755)
    p = _run(SCRIPT, {"PATH": f"{fake}:/usr/bin:/bin"})
    assert p.returncode == 0, p.stderr[-300:]
    out = p.stdout.strip()
    assert out.startswith("uvx ruff"), out
    assert "0.14" in out, "the fallback must pin a version: two ruffs can disagree about the same file"


def test_no_linter_anywhere_is_a_failure_with_an_install_command(tmp_path):
    # /usr/bin:/bin keeps bash resolvable (subprocess.execvp uses the env we hand it) while
    # putting neither ruff nor uvx on the path
    p = _run(SCRIPT, {"PATH": "/usr/bin:/bin"})
    assert p.returncode != 0, "silently succeeding here is exactly the trux bug"
    assert "uv tool install ruff" in p.stderr, p.stderr
