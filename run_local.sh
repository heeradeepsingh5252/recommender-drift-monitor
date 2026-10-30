#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [ ! -x .venv/bin/python ]; then
  echo "Virtual environment missing. Run ./bootstrap.sh first." >&2
  exit 1
fi
source .venv/bin/activate
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
export API_BASE_URL="${API_BASE_URL:-http://${API_HOST}:${API_PORT}}"
python -m uvicorn recommender_drift_monitor.api:app --app-dir src --host "$API_HOST" --port "$API_PORT" &
API_PID=$!
cleanup() {
  kill "$API_PID" 2>/dev/null || true
  wait "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
for attempt in $(seq 1 40); do
  if curl -fsS "$API_BASE_URL/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "API process exited early. See the error above." >&2
    exit 1
  fi
  sleep 0.5
done
if ! curl -fsS "$API_BASE_URL/health" >/dev/null 2>&1; then
  echo "API did not become healthy at $API_BASE_URL" >&2
  exit 1
fi
echo "API ready: $API_BASE_URL"
echo "API docs: $API_BASE_URL/docs"
echo "Opening Streamlit dashboard; press Ctrl+C to stop both processes."
python -m streamlit run dashboard.py --server.headless true
