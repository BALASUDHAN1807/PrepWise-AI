#!/usr/bin/env bash
# Starts the FastAPI backend (internal, :8000) and the Streamlit UI (public, :7860) in one container.
set -euo pipefail
export BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:8000}"

uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT

# wait for the API so the UI never shows "API offline" on first load
for _ in $(seq 1 60); do
  if curl -fs "$BACKEND_URL/health" >/dev/null 2>&1; then break; fi
  sleep 1
done

exec streamlit run frontend/app.py --server.port="${PORT:-7860}" --server.address=0.0.0.0 \
     --server.headless=true --browser.gatherUsageStats=false
