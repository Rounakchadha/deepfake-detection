#!/bin/bash
# start.sh — One command to launch Deepfake Detection (backend + frontend)
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "=== Deepfake Detection System ==="

# Kill any stale processes
pkill -f "uvicorn backend.api" 2>/dev/null && echo "Stopped old backend" || true
pkill -f "vite" 2>/dev/null && echo "Stopped old frontend" || true
sleep 1

# ── Backend ──────────────────────────────────────────────────────────────────
echo ""
echo "[1/2] Starting backend (FastAPI)..."
cd "$ROOT"
source venv/bin/activate

# Pre-warm: ensure HuggingFace models are cached locally
export HF_HOME="$ROOT/.hf_cache"
export TRANSFORMERS_OFFLINE=0   # allow first-time download if needed
export PYTHONPATH="$ROOT"

nohup env PYTHONPATH="$ROOT" HF_HOME="$ROOT/.hf_cache" \
  "$ROOT/venv/bin/python3" -m uvicorn backend.api:app \
    --host 0.0.0.0 --port 8000 --workers 1 \
    > "$ROOT/backend.log" 2>&1 &
BACKEND_PID=$!
echo "  Backend PID: $BACKEND_PID"

# Wait until backend is healthy (up to 60s to allow model loading)
echo "  Waiting for backend to be ready..."
for i in $(seq 1 60); do
  if curl -sf http://localhost:8000/ > /dev/null 2>&1; then
    echo "  Backend ready!"
    break
  fi
  if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "  ERROR: Backend crashed. Check backend.log:"
    tail -20 "$ROOT/backend.log"
    exit 1
  fi
  sleep 1
done

# ── Frontend ─────────────────────────────────────────────────────────────────
echo ""
echo "[2/2] Starting frontend (React + Vite)..."
cd "$ROOT/frontend-react"
nohup npm run dev > "$ROOT/frontend.log" 2>&1 &
FRONTEND_PID=$!
echo "  Frontend PID: $FRONTEND_PID"

sleep 3
echo ""
echo "========================================"
echo "  App ready at  →  http://localhost:3000"
echo "  API ready at  →  http://localhost:8000"
echo "========================================"
echo ""
echo "Logs: backend.log | frontend.log"
echo "Stop: pkill -f 'uvicorn backend.api'; pkill -f vite"
