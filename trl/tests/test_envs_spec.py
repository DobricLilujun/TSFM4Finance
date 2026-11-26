"""envs/envs.toml is executable configuration, not prose.

It names the python version and lock file for every family env, and `make_envs.sh` parses it
with tomllib. Duplicate keys make that a hard TOMLDecodeError, and because nothing imported
the file, the breakage sat unnoticed across three passes: the notes I appended for the lock
audit produced three `[env.x]` tables with two `notes =` each, so `make_envs.sh` died before
creating any venv. Parse it, and check the files it points at exist.
"""
from __future__ import annotations

import pathlib
import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPEC = ROOT / "envs" / "envs.toml"
ENVS_DIR = ROOT / "envs"


def test_envs_toml_parses():
    data = tomllib.loads(SPEC.read_text())
    assert data.get("env"), "expected [env.<name]> tables"
    assert len(data["env"]) >= 10, f"only {len(data['env'])} envs declared"


def test_every_env_names_a_python_and_a_source():
    for name, spec in tomllib.loads(SPEC.read_text())["env"].items():
        assert spec.get("python"), f"{name}: no python version"
        assert spec.get("lock") or spec.get("deps"), f"{name}: neither lock nor deps"


def test_referenced_lock_files_exist():
    missing = []
    for name, spec in tomllib.loads(SPEC.read_text())["env"].items():
        lock = spec.get("lock")
        if lock and not (ENVS_DIR / lock).exists():
            missing.append(f"{name}: {lock}")
    assert not missing, f"envs.toml references locks that are not in the repo: {missing}"


def test_no_duplicate_keys_per_table():
    """tomllib raises on duplicate keys, so this is really a readable failure message."""
    try:
        tomllib.loads(SPEC.read_text())
    except tomllib.TOMLDecodeError as e:  # pragma: no cover - the point is the message
        raise AssertionError(f"envs/envs.toml is invalid TOML ({e}); a duplicate key in a "
                             "[env.x] table breaks scripts/make_envs.sh") from e
