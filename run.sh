#!/usr/bin/env bash
# Finance TSFMs Arena — 一键本地启动 (后端 API + 前端网站, 同一端口 8000)
set -e
cd "$(dirname "$0")"
echo "[arena] 确保依赖已安装..."
uv venv .venv >/dev/null 2>&1 || true
uv pip install "pandas>=2.2" "numpy>=1.26" "pydantic>=2.7" "fastapi>=0.110" \
  "uvicorn[standard]>=0.29" "requests>=2.31" "scikit-learn>=1.4" "yfinance>=0.4" \
  "pyarrow" "pmdarima>=2.0" "chronos-forecasting" >/dev/null 2>&1 || true
echo "[arena] 启动服务: http://localhost:8000/"
uv run python -m backend.app
