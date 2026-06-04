#!/usr/bin/env bash
set -euo pipefail

SERVER_URL="${1:-http://172.16.100.26:10098}"
AUDIO_FILE="${2:-Fun-ASR-Nano-2512-Deploy/example.wav}"
MODE="${MODE:-2pass}"

if [[ ! -f "${AUDIO_FILE}" ]]; then
  echo "Audio file not found: ${AUDIO_FILE}" >&2
  exit 1
fi

echo "== Health =="
curl -fsS "${SERVER_URL}/health"
printf '\n\n'

echo "== File SSE ASR =="
echo "Server: ${SERVER_URL}"
echo "Audio:  ${AUDIO_FILE}"
curl -N --max-time 180 \
  -X POST "${SERVER_URL}/asr/file-sse" \
  -F "file=@${AUDIO_FILE}" \
  -F "mode=${MODE}" \
  -F "audio_fs=16000"
printf '\n'
