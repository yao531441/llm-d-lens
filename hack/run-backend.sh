#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/hack/storage-env.sh"
if [[ -x "${PYTHON_BIN:-}" ]]; then
  PYTHON="${PYTHON_BIN}"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON="$ROOT_DIR/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi
if ! "$PYTHON" -c 'import uvicorn' >/dev/null 2>&1; then
  echo "Backend dependency missing: uvicorn is not available in $PYTHON" >&2
  echo "Create a virtualenv and install the project requirements before starting Prism." >&2
  exit 1
fi
"$PYTHON" -m llm_d_lens.db.bootstrap_cli
BACKEND_PORT="${BACKEND_PORT:-${SERVER_PORT:-${PORT:-3000}}}"
SCHEME=http
TLS_ARGS=()
if [[ -n "${TLS_CERT_FILE:-}" && -n "${TLS_KEY_FILE:-}" ]]; then
  SCHEME=https
  export PRISM_MCP_CA_FILE="${PRISM_MCP_CA_FILE:-$TLS_CERT_FILE}"
  TLS_ARGS=(--ssl-certfile "$TLS_CERT_FILE" --ssl-keyfile "$TLS_KEY_FILE")
  if [[ -n "${TLS_CA_FILE:-}" ]]; then
    TLS_ARGS+=(--ssl-ca-certs "$TLS_CA_FILE")
  fi
fi
export PRISM_MCP_URL="${PRISM_MCP_URL:-$SCHEME://127.0.0.1:$BACKEND_PORT/api/assistant/mcp}"
exec "$PYTHON" -m uvicorn llm_d_lens.api:app \
  --host "${BACKEND_HOST:-${SERVER_HOST:-${HOST:-0.0.0.0}}}" \
  --port "$BACKEND_PORT" ${TLS_ARGS[@]+"${TLS_ARGS[@]}"}
