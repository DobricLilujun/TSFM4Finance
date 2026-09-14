#!/usr/bin/env bash
# Finance TSFMs Arena — local development server (FastAPI backend + static frontend on port 8000)
set -e

# Resolve project root (this script lives in scripts/)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "[arena] Ensuring virtual environment..."
uv venv .venv >/dev/null 2>&1 || true

echo "[arena] Installing tsfm_eval + tsfm4finance (editable)..."
uv pip install -e "src/tsfm_eval" -e "." >/dev/null 2>&1 || true

echo "[arena] Starting server: http://localhost:8000/"
uv run python -m tsfm4finance.backend.app
