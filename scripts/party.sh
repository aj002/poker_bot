#!/usr/bin/env bash
# Advisor server: builds the web app once and serves it on every network interface, so a phone on the same
# Wi-Fi can open the advisor page. The API stays on loopback behind the same port.
# Advisor sessions are saved under artifacts/advisor, so stopping and restarting this script loses nothing.
set -euo pipefail
cd "$(dirname "$0")/.."

API_PORT="${ARENA_API_PORT:-8317}"
WEB_PORT="${ARENA_WEB_PORT:-5317}"
UV="${UV:-uv}"

if ! command -v "$UV" >/dev/null 2>&1; then
  echo "uv is not installed. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh  (then open a new Terminal window)" >&2
  exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
  echo "Node.js is not installed. Get the LTS installer from https://nodejs.org (then open a new Terminal window)" >&2
  exit 1
fi
for port in "$API_PORT" "$WEB_PORT"; do
  if python3 -c "import socket,sys; s=socket.socket(); s.settimeout(0.3); sys.exit(0 if s.connect_ex(('127.0.0.1', $port))==0 else 1)"; then
    echo "Port $port is already in use. Is another 'make party' or 'make dev' still running? Stop it first (Ctrl-C in its window)." >&2
    exit 1
  fi
done

[ -d web/node_modules ] || (cd web && npm ci)
echo "Building the table (about 10 seconds)..."
(cd web && npm run build >/dev/null)

URLS="$("$UV" run python -m arena.api.party)"
if [ -z "$URLS" ]; then
  echo "Could not find this Mac's Wi-Fi address. Connect to Wi-Fi, or set ARENA_PUBLIC_URL=http://<address>:$WEB_PORT and run again." >&2
  exit 1
fi
FIRST="$(printf '%s\n' "$URLS" | head -n 1)"
export ARENA_PUBLIC_URL="${ARENA_PUBLIC_URL:-$FIRST}"
export ARENA_WEB_PORT="$WEB_PORT"

if [ -z "${ARENA_OPERATOR_TOKEN:-}" ]; then
  ARENA_OPERATOR_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')"
fi
export ARENA_OPERATOR_TOKEN
mkdir -p artifacts
(umask 077 && printf '%s\n' "$ARENA_OPERATOR_TOKEN" > artifacts/operator-token)

trap 'kill 0' EXIT INT TERM

"$UV" run uvicorn --factory arena.api.app:create_app --host 127.0.0.1 --port "$API_PORT" --log-level warning &
for _ in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1 && break
  sleep 0.5
done
(cd web && ARENA_API_PORT="$API_PORT" ARENA_WEB_PORT="$WEB_PORT" npx vite preview --host 0.0.0.0 --port "$WEB_PORT" --strictPort >/dev/null) &
for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$WEB_PORT/api/health" >/dev/null 2>&1 && break
  sleep 0.5
done

cat <<EOF

  Advisor is running.

  On this Mac, open:        $ARENA_PUBLIC_URL/#/advisor
  On your phone (same Wi-Fi): $ARENA_PUBLIC_URL/#/advisor
EOF
if [ "$(printf '%s\n' "$URLS" | wc -l)" -gt 1 ]; then
  echo "  Other addresses of this Mac (use one only if the first does not work):"
  printf '%s\n' "$URLS" | tail -n +2 | sed 's/^/                            /'
fi
echo "  Stop with Ctrl-C. Your session is saved, so starting again picks up where you left off."
echo
if [ "${ARENA_NO_OPEN:-0}" != "1" ] && command -v open >/dev/null 2>&1; then
  open "$ARENA_PUBLIC_URL/#/advisor" || true
fi
wait
