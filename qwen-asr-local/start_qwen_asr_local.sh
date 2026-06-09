#!/usr/bin/env bash
set -euo pipefail

exec python3 /app/qwen_asr_local_server.py \
  --host "${QWEN_ASR_HOST:-0.0.0.0}" \
  --port "${QWEN_ASR_PORT:-10100}"
