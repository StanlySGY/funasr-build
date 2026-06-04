#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-10097}"
BACKEND="${FUNASR_BACKEND_WS:-ws://127.0.0.1:10095}"

cd "${SCRIPT_DIR}"
python3 asr_sse_adapter.py --host "${HOST}" --port "${PORT}" --backend "${BACKEND}"
