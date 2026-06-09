#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE=${COMPOSE_FILE:-docker-compose.full.cpu.server.yml}
OVERRIDE_FILE=${OVERRIDE_FILE:-docker-compose.qwen-asr-local.yml}

mkdir -p Fun-ASR-Nano-2512-Deploy/asr_logs
mkdir -p /data/maas/sgy_arm/qwen-asr-models

docker compose -f "$COMPOSE_FILE" -f "$OVERRIDE_FILE" up -d --build qwen-asr-local funasr-sse-adapter

echo "Qwen-ASR local health: http://127.0.0.1:10100/health"
echo "Adapter health: http://127.0.0.1:10098/health"
echo "Test endpoint: POST http://127.0.0.1:10098/qwen-asr/file-sse"
