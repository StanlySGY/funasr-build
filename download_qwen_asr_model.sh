#!/usr/bin/env bash
set -euo pipefail

MODEL_ID=${MODEL_ID:-Qwen/Qwen3-ASR-0.6B}
MODEL_DIR=${MODEL_DIR:-/data/maas/sgy_arm/qwen-asr-models/Qwen3-ASR-0.6B}
CACHE_DIR=${CACHE_DIR:-/data/maas/sgy_arm/qwen-asr-models/modelscope}
IMAGE=${IMAGE:-qwen-asr-local:latest}

echo "model_id=${MODEL_ID}"
echo "model_dir=${MODEL_DIR}"
echo "cache_dir=${CACHE_DIR}"
mkdir -p "$MODEL_DIR" "$CACHE_DIR"

docker run --rm --network host \
  -e MODEL_ID="$MODEL_ID" \
  -e MODELSCOPE_CACHE=/models/modelscope \
  -v /data/maas/sgy_arm/qwen-asr-models:/models \
  "$IMAGE" \
  python3 - <<'PY'
import os
import shutil
from pathlib import Path
from modelscope import snapshot_download

model_id = os.environ.get("MODEL_ID", "Qwen/Qwen3-ASR-0.6B")
model_dir = Path("/models/Qwen3-ASR-0.6B")
cache_dir = "/models/modelscope"
model_dir.mkdir(parents=True, exist_ok=True)

try:
    path = snapshot_download(model_id, cache_dir=cache_dir, local_dir=str(model_dir))
except TypeError:
    path = snapshot_download(model_id, cache_dir=cache_dir)
    source = Path(path)
    for item in source.iterdir():
        target = model_dir / item.name
        if target.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)

print(f"downloaded_to={model_dir}")
PY

echo "downloaded files:"
find "$MODEL_DIR" -maxdepth 2 -type f | sed -n '1,40p'
