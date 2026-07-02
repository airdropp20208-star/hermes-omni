#!/bin/bash
# Hermes-Omni Dashboard launcher
# Works on UserLAnd (Android), Linux, macOS
#
# Usage:
#   ./start.sh              # default port 8788
#   ./start.sh 9000         # custom port
#   HERMES_API_KEY=sk-... ./start.sh   # set key via env

set -e
cd "$(dirname "$0")"

# ─── Config ────────────────────────────────────────────────────────────
PORT="${1:-8788}"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
PYTHON="${PYTHON:-python3}"

mkdir -p "$HERMES_HOME/uploads" "$HERMES_HOME/unified" "$HERMES_HOME/logs"

# ─── Detect Python ─────────────────────────────────────────────────────
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    PYTHON="python"
fi
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "❌ Không tìm thấy Python. Cài đặt Python 3.10+ trước."
    exit 1
fi

PY_VER=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "🐍 Python: $PY_VER ($($PYTHON -c 'import sys; print(sys.executable)'))"

# ─── Install critical deps if missing ──────────────────────────────────
echo "📦 Kiểm tra dependencies..."
$PYTHON -c "import yaml" 2>/dev/null || $PYTHON -m pip install --user --quiet pyyaml
$PYTHON -c "import openai" 2>/dev/null || $PYTHON -m pip install --user --quiet openai
$PYTHON -c "import httpx" 2>/dev/null || $PYTHON -m pip install --user --quiet httpx
$PYTHON -c "import requests" 2>/dev/null || $PYTHON -m pip install --user --quiet requests

# ─── If API key provided via env, write to .env ────────────────────────
if [ -n "$HERMES_API_KEY" ]; then
    ENV_FILE="$HERMES_HOME/.env"
    # Remove existing XIAOMI_API_KEY and ZAI_API_KEY lines
    if [ -f "$ENV_FILE" ]; then
        grep -v "^XIAOMI_API_KEY=" "$ENV_FILE" | grep -v "^ZAI_API_KEY=" > "$ENV_FILE.tmp" 2>/dev/null || true
        mv "$ENV_FILE.tmp" "$ENV_FILE" 2>/dev/null || true
    fi
    echo "XIAOMI_API_KEY=$HERMES_API_KEY" >> "$ENV_FILE"
    echo "HERMES_INFERENCE_PROVIDER=xiaomi" >> "$ENV_FILE"
    echo "HERMES_INFERENCE_MODEL=mimo-v2.5" >> "$ENV_FILE"
    echo "✅ Đã lưu API key vào $ENV_FILE"
fi

# ─── Kill any previous dashboard ───────────────────────────────────────
pkill -f "dashboard_server" 2>/dev/null || true
sleep 1

# ─── Launch ────────────────────────────────────────────────────────────
echo ""
echo "🚀 Khởi động Hermes-Omni Dashboard v6..."
echo "   Port:         $PORT"
echo "   Bind:         127.0.0.1 (chỉ localhost — an toàn)"
echo "   Hermes home:  $HERMES_HOME"
echo "   URL:          http://localhost:$PORT"
echo "   Token file:   $HERMES_HOME/.dashboard_token"
echo "   ⚠ Truy cập ngoài localhost: dùng SSH tunnel"
echo ""

export HERMES_HOME
exec "$PYTHON" -m agent.unified.dashboard_server --port "$PORT" --host 127.0.0.1
