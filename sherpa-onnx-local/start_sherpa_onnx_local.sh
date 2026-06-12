#!/usr/bin/env bash
set -euo pipefail

exec python3 /app/sherpa_onnx_local_server.py \
  --host "${SHERPA_ONNX_HOST:-0.0.0.0}" \
  --port "${SHERPA_ONNX_PORT:-10110}"
