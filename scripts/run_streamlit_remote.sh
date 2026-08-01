#!/usr/bin/env bash
set -euo pipefail

HOST="${FDM_STREAMLIT_HOST:-127.0.0.1}"
PORT="${FDM_STREAMLIT_PORT:-8501}"

exec uv run python -m streamlit run ui/app.py \
  --server.address "$HOST" \
  --server.port "$PORT" \
  --server.headless true \
  --browser.gatherUsageStats false \
  "$@"
