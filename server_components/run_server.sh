#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/work/VALL-E"
cd "$PROJECT_ROOT"

# ────────────────────────────────────────────────
# 1. Install direnv if missing
# ────────────────────────────────────────────────
if ! command -v direnv >/dev/null 2>&1; then
  echo "[+] direnv not found, installing..."
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y && sudo apt-get install -y direnv
  elif command -v yum >/dev/null 2>&1; then
    sudo yum install -y direnv
  else
    echo "[!] Package manager not found. Please install direnv manually."
    exit 1
  fi
else
  echo "[✓] direnv already installed."
fi

# ────────────────────────────────────────────────
# 2. Authorize direnv to load .envrc
# ────────────────────────────────────────────────
if [ -f ".envrc" ]; then
  echo "[+] Allowing .envrc in $PROJECT_ROOT"
  direnv allow "$PROJECT_ROOT"
else
  echo "[!] No .envrc found in $PROJECT_ROOT — skipping direnv allow."
fi

# ────────────────────────────────────────────────
# 3. Load environment variables from direnv
# ────────────────────────────────────────────────
eval "$(direnv export bash || true)"

# ────────────────────────────────────────────────
# 4. Start Ngrok and Uvicorn
# ────────────────────────────────────────────────
NGROK_BIN=${NGROK_BIN:-ngrok}
NGROK_PORT=${NGROK_PORT:-8000}
NGROK_FIXED_DOMAIN=${NGROK_FIXED_DOMAIN:-honest-trivially-buffalo.ngrok-free.app}
NGROK_LOG=${NGROK_LOG:-"$PROJECT_ROOT/ngrok.log"}

# stop old ngrok if running
if pgrep -x "ngrok" >/dev/null 2>&1; then
  echo "[+] Stopping existing ngrok process..."
  pkill ngrok || true
  sleep 1
fi

echo "[+] Starting ngrok..."
$NGROK_BIN http "$NGROK_PORT" --url="${NGROK_FIXED_DOMAIN}" >"$NGROK_LOG" 2>&1 &

sleep 3
echo "[+] ngrok log (last 20 lines):"
tail -n 20 "$NGROK_LOG" || true

echo "[+] Starting uvicorn..."
uvicorn app:app --host 0.0.0.0 --port "$NGROK_PORT" --workers 1
