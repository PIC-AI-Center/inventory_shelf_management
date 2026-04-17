#!/usr/bin/env bash
# Start the FastAPI server and Celery worker locally (no Docker).
# Prerequisites:
#   1. pip install -r requirements.txt
#   2. Copy .env.example -> .env and fill in your values
#   3. Redis running at REDIS_URL (default: redis://localhost:6379/0)
#   4. Triton running at TRITON_HTTP_HOST:TRITON_HTTP_PORT and :TRITON_GRPC_PORT

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f .env ]; then
  echo "ERROR: .env not found. Copy .env.example -> .env and fill in your values."
  exit 1
fi

# Load env
export $(grep -v '^#' .env | xargs)

FASTAPI_PORT="${FASTAPI_PORT:-8082}"
FASTAPI_WORKERS="${FASTAPI_WORKERS:-4}"
CELERY_CONCURRENCY="${CELERY_WORKER_CONCURRENCY:-4}"

# Ensure session dir exists
mkdir -p "${REF_CACHE_PATH:-request_case}/match_sessions"
mkdir -p "${PLANOGRAM_CSV_DIR:-planograms}"

echo "==> Starting Celery worker (concurrency=${CELERY_CONCURRENCY})..."
celery -A celery_app worker \
  --loglevel=info \
  --concurrency="${CELERY_CONCURRENCY}" \
  --queues=celery \
  &
CELERY_PID=$!

echo "==> Starting FastAPI on port ${FASTAPI_PORT} (workers=${FASTAPI_WORKERS})..."
uvicorn main:app \
  --host 0.0.0.0 \
  --port "${FASTAPI_PORT}" \
  --workers "${FASTAPI_WORKERS}" \
  --log-level info \
  &
API_PID=$!

echo ""
echo "  API  : http://localhost:${FASTAPI_PORT}"
echo "  Docs : http://localhost:${FASTAPI_PORT}/docs"
echo ""
echo "Press Ctrl+C to stop both processes."

cleanup() {
  echo "Stopping..."
  kill "$CELERY_PID" "$API_PID" 2>/dev/null || true
  wait "$CELERY_PID" "$API_PID" 2>/dev/null || true
}
trap cleanup INT TERM

wait
