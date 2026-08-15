"""Encrypted configuration store shared between the management server and the arena.

The enabled-datasets / enabled-models configuration is persisted at rest ENCRYPTED
with Fernet (AES-128-CBC + HMAC). The symmetric key is derived from an admin
passphrase via PBKDF2-HMAC-SHA256, so `config.enc` on disk is unreadable without
the passphrase.

The management server (authenticated) is the only writer. On each write it also
emits a plaintext read-mirror (`config.json`) that the public arena reads quickly
without holding any secret. If the mirror is absent, everything defaults to enabled.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
CONFIG_DIR = DATA_DIR / "config"
CONFIG_FILE = CONFIG_DIR / "config.enc"      # encrypted blob (source of truth)
MIRROR_FILE = CONFIG_DIR / "config.json"     # plaintext read-mirror for the arena
METADATA_FILE = CONFIG_DIR / "config.meta"   # salt + version (not secret)


def _derive_key(passphrase: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=200_000)
    return base64.urlsafe_b64encode(kdf.derive(passphrase.encode("utf-8")))


DEFAULT_CONFIG: dict[str, Any] = {"datasets": True, "models": True}


# ---------------------------------------------------------------------------
# Encrypted source of truth
# ---------------------------------------------------------------------------
def _load_metadata() -> dict:
    """Return metadata, creating AND persisting the salt file on first use so the
    salt (and thus the derived key) is stable across read/write calls."""
    if METADATA_FILE.exists():
        try:
            return json.loads(METADATA_FILE.read_text())
        except Exception:
            pass
    meta = {"salt": base64.urlsafe_b64encode(os.urandom(16)).decode(), "version": 1}
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        METADATA_FILE.write_text(json.dumps(meta))
    except Exception:
        pass
    return meta


def _fernet(passphrase: str) -> Fernet:
    salt = _load_metadata()["salt"]
    return Fernet(_derive_key(passphrase, base64.urlsafe_b64decode(salt.encode())))


def read_encrypted(passphrase: str) -> dict:
    """Read the config by decrypting `config.enc`. Wrong passphrase => defaults."""
    if not CONFIG_FILE.exists():
        return DEFAULT_CONFIG.copy()
    try:
        token = CONFIG_FILE.read_bytes()
        payload = _fernet(passphrase).decrypt(token)
        data = json.loads(payload.decode("utf-8"))
        return data if isinstance(data, dict) else DEFAULT_CONFIG.copy()
    except (InvalidToken, json.JSONDecodeError, ValueError):
        return DEFAULT_CONFIG.copy()


def write_encrypted(data: dict, passphrase: str) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    salt = _load_metadata()["salt"]
    payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
    CONFIG_FILE.write_bytes(_fernet(passphrase).encrypt(payload))
    meta = _load_metadata()
    meta["version"] = meta.get("version", 1) + 1
    METADATA_FILE.write_text(json.dumps(meta))
    _write_mirror(data)


def _write_mirror(data: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    MIRROR_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Public arena reads the plaintext mirror (no secret needed)
# ---------------------------------------------------------------------------
def current() -> dict:
    if MIRROR_FILE.exists():
        try:
            return json.loads(MIRROR_FILE.read_text())
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()


def _enabled_map(data: dict, key: str) -> dict | None:
    v = data.get(key, True)
    if v is True:
        return None
    if isinstance(v, dict):
        return v
    return None


def _is_enabled(m: dict | None, name: str) -> bool:
    if m is None:
        return True
    return m.get(name, True)


def is_dataset_enabled(name: str) -> bool:
    return _is_enabled(_enabled_map(current(), "datasets"), name)


def is_model_enabled(name: str) -> bool:
    return _is_enabled(_enabled_map(current(), "models"), name)


def enabled_datasets() -> set[str] | None:
    m = _enabled_map(current(), "datasets")
    return None if m is None else {k for k, v in m.items() if v}


def enabled_models() -> set[str] | None:
    m = _enabled_map(current(), "models")
    return None if m is None else {k for k, v in m.items() if v}