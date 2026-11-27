"""Third-party source checkouts for the families that publish no wheel.

Sundial, Kronos, VisionTS++ and friends ship as *repositories* (sometimes with the
weights on the Hub but the code only in git). Rather than every adapter inventing its
own ``sys.path`` hack, they call :func:`ensure_on_path`, which adds:

1. every directory in ``$TSFMTRL_THIRD_PARTY`` (``os.pathsep``-separated),
2. every subdirectory of ``$TSFMTRL_THIRD_PARTY/third_party``,
3. ``<cwd>/third_party/*`` and ``<cwd>/envs/*``,

and returns what it added, so an import failure can name the missing checkout instead
of pretending the model is broken.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ADDED: list[str] = []


def candidate_dirs() -> list[Path]:
    out: list[Path] = []
    root = os.environ.get("TSFMTRL_THIRD_PARTY", "")
    for p in filter(None, root.split(os.pathsep)):
        q = Path(p).expanduser()
        out.append(q)
        if q.is_dir():
            out += [d for d in sorted(q.iterdir()) if d.is_dir()]
    for base in (Path.cwd() / "third_party", Path.cwd() / "envs"):
        if base.is_dir():
            out += [d for d in sorted(base.iterdir()) if d.is_dir()]
    return out


def ensure_on_path(extra: "str | list[str] | None" = None) -> list[str]:
    """Put the third-party checkouts on ``sys.path``. Returns the entries added."""
    dirs: list[Path] = []
    if extra:
        dirs += [Path(p).expanduser() for p in (extra if isinstance(extra, list) else [extra])]
    dirs += candidate_dirs()
    added = []
    for d in dirs:
        s = str(d)
        if d.is_dir() and s not in sys.path:
            sys.path.insert(0, s)
            added.append(s)
    _ADDED.extend(added)
    return added


def hint(name: str) -> str:
    """The install line an error message should carry."""
    return (f"clone it and point the library at it: export TSFMTRL_THIRD_PARTY="
            f"$HOME/third_party/{name}  (currently: {os.environ.get('TSFMTRL_THIRD_PARTY', 'unset')})")
