"""`scripts/check.sh` must say which interpreter it gated, and prefer a pinned env.

The harness pins per-family environments from lock files, and the lock file names the interpreter that
produced them. A green gate on an arbitrary `python3` is a different claim from a green gate on
`envs/core`, so the choice is printed - and an explicitly requested interpreter (TSFMTRL_PY) wins over
whatever happens to be in the tree.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "check.sh"


def test_the_interpreter_line_names_the_python_that_ran(tmp_path):
    # a stub interpreter: the gate will fail elsewhere, but the interpreter line must be printed first
    stub = tmp_path / "python"
    stub.write_text('#!/bin/sh\necho "3.11.0"\n')
    stub.chmod(0o755)
    p = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, timeout=300,
                       env={**os.environ, "TSFMTRL_PY": str(stub),
                            # stop at the first gate so the test stays cheap
                            "PY": str(stub)})
    assert "interpreter:" in p.stdout, p.stdout[:400]
    assert str(stub) in p.stdout, "the requested interpreter must be the one named"
