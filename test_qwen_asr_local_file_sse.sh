#!/usr/bin/env bash
set -euo pipefail

AUDIO_FILE=${1:-example.wav}
BASE_URL=${BASE_URL:-http://127.0.0.1:10098}

curl -N -X POST "$BASE_URL/qwen-asr/file-sse" \
  -F "file=@${AUDIO_FILE}" \
  -F "mode=online" \
  -F "audio_fs=16000" \
  -F "hotwords="
