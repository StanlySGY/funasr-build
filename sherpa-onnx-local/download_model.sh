#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${SHERPA_ONNX_HOST_MODEL_DIR:-${SHERPA_ONNX_MODEL_DIR:-/data/maas/sgy_arm/sherpa-onnx-models}}"
MODEL_URL="${SHERPA_ONNX_MODEL_URL:-https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2}"
WORK_DIR="${TMPDIR:-/tmp}/sherpa-onnx-model-download"
ARCHIVE="${WORK_DIR}/${MODEL_URL##*/}"

mkdir -p "${MODEL_DIR}" "${WORK_DIR}"

if [ ! -w "${MODEL_DIR}" ]; then
  echo "No write permission for model dir: ${MODEL_DIR}" >&2
  echo "Either fix ownership, or set SHERPA_ONNX_HOST_MODEL_DIR to a writable directory." >&2
  exit 1
fi

if [ -f "${MODEL_DIR}/tokens.txt" ] && { [ -f "${MODEL_DIR}/model.int8.onnx" ] || [ -f "${MODEL_DIR}/model.onnx" ]; }; then
  echo "sherpa-onnx model already exists in ${MODEL_DIR}"
  ls -lh "${MODEL_DIR}"
  exit 0
fi

echo "Downloading sherpa-onnx model:"
echo "  url: ${MODEL_URL}"
echo "  dir: ${MODEL_DIR}"

if [ -f "${ARCHIVE}" ]; then
  echo "Using existing archive: ${ARCHIVE}"
elif command -v curl >/dev/null 2>&1; then
  curl -L --fail --retry 3 -o "${ARCHIVE}" "${MODEL_URL}"
elif command -v wget >/dev/null 2>&1; then
  wget -O "${ARCHIVE}" "${MODEL_URL}"
else
  echo "curl or wget is required" >&2
  exit 1
fi

rm -rf "${WORK_DIR}/extract"
mkdir -p "${WORK_DIR}/extract"
case "${ARCHIVE}" in
  *.tar.bz2|*.tbz2)
    tar -xjf "${ARCHIVE}" -C "${WORK_DIR}/extract"
    ;;
  *.tar.gz|*.tgz)
    tar -xzf "${ARCHIVE}" -C "${WORK_DIR}/extract"
    ;;
  *.zip)
    unzip -q "${ARCHIVE}" -d "${WORK_DIR}/extract"
    ;;
  *)
    echo "Unsupported archive format: ${ARCHIVE}" >&2
    exit 1
    ;;
esac

TOKENS_PATH="$(find "${WORK_DIR}/extract" -type f -name tokens.txt | head -n 1)"
MODEL_PATH="$(find "${WORK_DIR}/extract" -type f \( -name model.int8.onnx -o -name model.onnx -o -name '*.onnx' \) | sort | head -n 1)"

if [ -z "${TOKENS_PATH}" ] || [ -z "${MODEL_PATH}" ]; then
  echo "Could not find tokens.txt and ONNX model in extracted archive" >&2
  find "${WORK_DIR}/extract" -maxdepth 3 -type f | sort >&2
  exit 1
fi

cp -f "${TOKENS_PATH}" "${MODEL_DIR}/tokens.txt"
if [ "$(basename "${MODEL_PATH}")" = "model.int8.onnx" ]; then
  cp -f "${MODEL_PATH}" "${MODEL_DIR}/model.int8.onnx"
else
  cp -f "${MODEL_PATH}" "${MODEL_DIR}/model.onnx"
fi

echo "Installed sherpa-onnx model files:"
ls -lh "${MODEL_DIR}"
