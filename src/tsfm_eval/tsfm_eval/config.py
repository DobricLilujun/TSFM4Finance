"""Runtime configuration for tsfm_eval (data root, environment variables)."""
from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_DATA_ROOT = Path.cwd() / "assets" / "data"

_data_root: Path | None = None


def get_data_root() -> Path:
    """Return the active data root directory.

    Resolution order:
      1. Value set by ``set_data_root()`` in this session.
      2. ``TSFM_EVAL_DATA_ROOT`` environment variable.
      3. ``./assets/data`` relative to the current working directory.
    """
    global _data_root
    if _data_root is not None:
        return _data_root
    env = os.environ.get("TSFM_EVAL_DATA_ROOT")
    if env:
        return Path(env)
    return _DEFAULT_DATA_ROOT


def set_data_root(path: str | Path) -> None:
    """Override the data root for the current process."""
    global _data_root
    _data_root = Path(path)
