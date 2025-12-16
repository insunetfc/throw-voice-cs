#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   NGROK_AUTHTOKEN=xxx NGROK_TTS_DOMAIN=honest-trivially-buffalo.ngrok-free.app ./create_ngrok_config.sh
#   (NGROK_TTS_DOMAIN is optional; omit it if you don't have a reserved domain)

NGROK_CFG_DIR="${HOME}/.config/ngrok"
NGROK_CFG="${NGROK_CFG_DIR}/ngrok.yml"

: "${NGROK_AUTHTOKEN:?Set NGROK_AUTHTOKEN env var}"
TTS_DOMAIN="${NGROK_TTS_DOMAIN:-}"   # optional

mkdir -p "${NGROK_CFG_DIR}"

# Backup existing config if present
if [[ -f "${NGROK_CFG}" ]]; then
  ts=$(date +%Y%m%d-%H%M%S)
  cp -f "${NGROK_CFG}" "${NGROK_CFG}.bak.${ts}"
  echo "Backed up existing config to ${NGROK_CFG}.bak.${ts}"
fi

# Write base config (v3 style)
cat > "${NGROK_CFG}" <<EOF
version: "3"
agent:
  authtoken: ${NGROK_AUTHTOKEN}
tunnels:
  tts:
    proto: http
    addr: 8000
EOF

# If you have a reserved domain, pin TTS to it
if [[ -n "${TTS_DOMAIN}" ]]; then
  cat >> "${NGROK_CFG}" <<EOF
    domain: ${TTS_DOMAIN}
EOF
fi

# Add Flask + WebSocket tunnels (random ngrok subdomains)
cat >> "${NGROK_CFG}" <<'EOF'
  flask:
    proto: http
    addr: 5000
  ws:
    proto: http     # ngrok serves both https:// and wss:// for this
    addr: 8765
EOF

echo "Wrote ${NGROK_CFG}"
echo
echo "Next steps:"
echo "  1) Stop any existing ngrok agent session (you can disconnect it in the dashboard)."
echo "  2) Start all tunnels from a single agent:"
echo "       ngrok start --all"
echo "     (or: ngrok start tts flask ws)"
echo
echo "After it starts, you'll see forwarding URLs like:"
echo "  - TTS (8000):  https://<...>.ngrok-free.app  (or your reserved domain)"
echo "  - Flask (5000): https://<...>.ngrok-free.app"
echo "  - WS (8765):    https://<...>.ngrok-free.app  (use as wss:// for Twilio)"
echo
echo "Twilio webhooks:"
echo "  Gather webhook → https://<flask-host>.ngrok-free.app/voice_gather"
echo "  Media Streams  → wss://<ws-host>.ngrok-free.app/twilio"
