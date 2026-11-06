"""Encrypted management server for Finance TSFMs Arena.

A small, authenticated admin API (separate from the public FastAPI backend) that
lets you toggle which datasets and models are "supported". The configuration is
stored ENCRYPTED at rest (Fernet / AES-128-CBC+HMAC, key from a PBKDF2-derived
passphrase) and mirrored to a plaintext file the public arena reads.

Auth: a single admin password, stored as a PBKDF2-HMAC-SHA256 hash (100k iters)
-> short-lived JWT session token. (No passlib/bcrypt: pure `cryptography`.)

Run:  uv run python -m backend.manage
Default admin:  user `admin`, password `ChangeMe123!`  (override with env
ADMIN_USER / ADMIN_PASSWORD before first run, or set them here).
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time

import jwt
from fastapi import Depends, FastAPI, HTTPException, Request

from backend.config import current, write_encrypted

ADMIN_USER = os.environ.get("ADMIN_USER", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "ChangeMe123!")
JWT_SECRET = os.environ.get("JWT_SECRET") or os.urandom(24).hex()
TOKEN_TTL = 8 * 3600   # 8 hours
HASH_ITERS = 100_000

app = FastAPI(title="Finance TSFMs Arena — Management", version="1.0.0")


# ---- password (PBKDF2, no third-party hash lib) --------------------------
def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, HASH_ITERS)


def _verify_password(password: str, stored: str) -> bool:
    """stored = 'salt:hash' (hex). Constant-time compare."""
    try:
        salt_hex, hash_hex = stored.split(":", 1)
        return hmac.compare_digest(
            _hash_password(password, bytes.fromhex(salt_hex)).hex(), hash_hex)
    except Exception:
        return False


# ---- auth -----------------------------------------------------------------
def _new_token() -> str:
    now = int(time.time())
    return jwt.encode({"sub": ADMIN_USER, "iat": now, "exp": now + TOKEN_TTL},
                      JWT_SECRET, algorithm="HS256")


def _verify(token: str) -> bool:
    try:
        jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return True
    except Exception:
        return False


def require_auth(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    token = auth[7:] if auth.lower().startswith("bearer ") else ""
    if not token or not _verify(token):
        raise HTTPException(401, "authentication required")
    return token


# ---- config helpers -------------------------------------------------------
def _all_names() -> dict:
    """Enumerate every dataset and model the arena knows about.

    Datasets are read from the on-disk meta.json files (the same robust source
    the public backend uses), NOT from build_all() — which re-runs generators and
    can fail/return 0 (e.g. parquet/closed datasets needing pyarrow)."""
    from pathlib import Path
    from arena import schemas as S

    data_dir = Path(__file__).resolve().parent.parent / "data"
    datasets: list[str] = []
    for bucket in ("open", "closed"):
        base = data_dir / bucket
        for d in base.glob("*") if base.exists() else []:
            p = d / "meta.json"
            if p.exists():
                try:
                    meta = S.DatasetMeta.model_validate_json(p.read_text())
                    datasets.append(meta.name)
                except Exception:
                    datasets.append(d.name)
    models: list[str] = []
    try:
        from arena import models as M
        for n in M.available_models():
            models.append(n)
    except Exception:
        pass
    return {"datasets": sorted(set(datasets)), "models": sorted(set(models))}


def _full_config() -> dict:
    """Config expanded to explicit per-name maps (missing names default True)."""
    names = _all_names()
    cur = current()
    ds = cur.get("datasets")
    md = cur.get("models")
    ds_map = ds if isinstance(ds, dict) else {}
    md_map = md if isinstance(md, dict) else {}
    return {
        "datasets": {n: bool(ds_map.get(n, True)) for n in names["datasets"]} or ds_map,
        "models": {n: bool(md_map.get(n, True)) for n in names["models"]} or md_map,
    }


# ---- routes ---------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "service": "management", "auth": "enabled"}


@app.post("/api/login")
def login(body: dict):
    if body.get("username", "") == ADMIN_USER and _verify_password(
            body.get("password", ""), _password_hash()):
        return {"token": _new_token(), "expires_in": TOKEN_TTL, "user": ADMIN_USER}
    raise HTTPException(401, "invalid credentials")


def _password_hash() -> str:
    """Hash the configured admin password with a fixed salt (deterministic)."""
    salt = hashlib.sha256(b"tsfm4finance-salt").digest()
    return f"{salt.hex()}:{_hash_password(ADMIN_PASSWORD, salt).hex()}"


@app.get("/api/config")
def get_config(_: str = Depends(require_auth)):
    return _full_config()


@app.post("/api/config")
def set_config(body: dict, _: str = Depends(require_auth)):
    """Replace the enabled map. body: {datasets: {...|bool}, models: {...|bool}}."""
    cur = _full_config()
    for key in ("datasets", "models"):
        if key not in body:
            continue
        val = body[key]
        if isinstance(val, bool):
            cur[key] = {n: val for n in cur.get(key, {})}
        elif isinstance(val, dict):
            merged = cur.get(key, {})
            merged.update({k: bool(v) for k, v in val.items()})
            cur[key] = merged
    write_encrypted(cur, ADMIN_PASSWORD)
    return {"ok": True, "config": _full_config()}


@app.post("/api/dataset/{name}/toggle")
def toggle_dataset(name: str, _: str = Depends(require_auth)):
    cfg = _full_config()
    cur = cfg["datasets"].get(name, True)
    cfg["datasets"][name] = not cur
    write_encrypted(cfg, ADMIN_PASSWORD)
    return {"ok": True, "dataset": name, "enabled": not cur}


@app.post("/api/model/{name}/toggle")
def toggle_model(name: str, _: str = Depends(require_auth)):
    cfg = _full_config()
    cfg["models"][name] = not cfg["models"].get(name, True)
    write_encrypted(cfg, ADMIN_PASSWORD)
    return {"ok": True, "model": name, "enabled": not cfg["models"][name]}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("MANAGE_PORT", "8100"))
    print(f"Encrypted management server on :{port}  (admin='{ADMIN_USER}')")
    uvicorn.run(app, host="0.0.0.0", port=port)