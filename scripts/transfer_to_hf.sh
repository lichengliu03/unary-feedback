#!/bin/bash
# Export a trained checkpoint to Hugging Face format and optionally upload it.
#
# Minimal usage:
#   CKPT_DIR=outputs/checkpoints/exp1_MetamathQA \
#   MODEL_PATH=Qwen/Qwen2.5-3B-Instruct \
#   bash scripts/transfer_to_hf.sh
#
# Upload to Hugging Face after running `hf auth login`:
#   HF_REPO_ID=your-name/exp1-metamathqa \
#   CKPT_DIR=outputs/checkpoints/exp1_MetamathQA \
#   MODEL_PATH=Qwen/Qwen2.5-3B-Instruct \
#   bash scripts/transfer_to_hf.sh

set -euo pipefail
set +x

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}"

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN:-python}"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN:-python3}"
else
  echo "[ERROR] Neither python nor python3 was found in PATH."
  exit 1
fi

CKPT_DIR="${CKPT_DIR:?Please set CKPT_DIR to the experiment dir, global_step dir, or actor dir}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
STEP="${STEP:-latest}"
WORLD_SIZE="${WORLD_SIZE:-${NGPUS:-}}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
HF_REPO_ID="${HF_REPO_ID:-}"
HF_PRIVATE_REPO="${HF_PRIVATE_REPO:-0}"

ARGS=(
  --input-dir "${CKPT_DIR}"
  --step "${STEP}"
  --base-model "${MODEL_PATH}"
)

if [[ -n "${WORLD_SIZE}" ]]; then
  ARGS+=(--world-size "${WORLD_SIZE}")
fi

if [[ -n "${OUTPUT_DIR}" ]]; then
  ARGS+=(--output-dir "${OUTPUT_DIR}")
fi

if [[ -n "${HF_REPO_ID}" ]]; then
  ARGS+=(--repo-id "${HF_REPO_ID}")
fi

if [[ "${HF_PRIVATE_REPO}" == "1" ]]; then
  ARGS+=(--private)
fi

echo "[INFO] ============================================"
echo "[INFO] Transfer to Hugging Face"
echo "[INFO] Python:      ${PYTHON_BIN}"
echo "[INFO] Checkpoint:  ${CKPT_DIR}"
echo "[INFO] Step:        ${STEP}"
echo "[INFO] Base model:  ${MODEL_PATH}"
echo "[INFO] World size:  ${WORLD_SIZE:-<auto>}"
echo "[INFO] Output dir:  ${OUTPUT_DIR:-<auto>}"
echo "[INFO] HF repo:     ${HF_REPO_ID:-<skip upload>}"
echo "[INFO] ============================================"

"${PYTHON_BIN}" scripts/utils/convert_fsdp_to_hf.py "${ARGS[@]}"
