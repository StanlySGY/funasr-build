#!/usr/bin/env bash
set -euo pipefail

AUDIO_FILE=${1:-example.wav}
BASE_URL=${BASE_URL:-http://127.0.0.1:10098}

if [ ! -f "$AUDIO_FILE" ]; then
  echo "audio file not found: $AUDIO_FILE" >&2
  echo "usage: $0 /path/to/audio.wav" >&2
  exit 2
fi

curl -N -X POST "$BASE_URL/qwen-asr/file-sse" \
  -F "file=@${AUDIO_FILE}" \
  -F "mode=online" \
  -F "audio_fs=16000" \
  -F "hotwords="
