#!/bin/bash
# Single-GPU quick test for any environment.
#
# Usage:
#   bash scripts/quick_test.sh                          # default: MetamathQA
#   ENV_TAG=Countdown bash scripts/quick_test.sh
#   ENV_TAG=SimpleSokoban bash scripts/quick_test.sh
#   ENV_TAG=FrozenLake bash scripts/quick_test.sh
#
# Available ENV_TAGs (from configs/envs.yaml):
#   MetamathQA, Countdown, SimpleSokoban, FrozenLake, Bandit,
#   HotpotQA, GSM8k, MATH, AIME24, WebShop, SimpleSudoku, ...

set -euo pipefail

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate ragen_ufb
fi

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}"

mkdir -p outputs/checkpoints
mkdir -p outputs/logs

export PYTHONPATH="${PROJECT_DIR}/verl:${PROJECT_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=ERROR
export VLLM_LOG_LEVEL=WARNING
export TRANSFORMERS_VERBOSITY=warning
export HF_HUB_VERBOSITY=error
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export HF_TOKEN="${HF_TOKEN:-}"

# ---- configurable ----
ENV_TAG="${ENV_TAG:-MetamathQA}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
STEPS="${STEPS:-5}"
EXPERIMENT="${EXPERIMENT:-test_1gpu_${ENV_TAG}}"
CKPT_DIR="${CKPT_DIR:-${PROJECT_DIR}/outputs/checkpoints/${EXPERIMENT}}"
LOG_FILE="${PROJECT_DIR}/outputs/logs/${ENV_TAG}.log"

mkdir -p "${CKPT_DIR}"

echo "[INFO] ============================================"
echo "[INFO] Single-GPU test"
echo "[INFO] Env tag:     ${ENV_TAG}"
echo "[INFO] Model:       ${MODEL_PATH}"
echo "[INFO] Steps:       ${STEPS}"
echo "[INFO] Checkpoint:  ${CKPT_DIR}"
echo "[INFO] Log file:    ${LOG_FILE}"
echo "[INFO] ============================================"

# Use base config directly to avoid Hydra defaults chain issues with envs/*.yaml
python train.py \
  --config-name=base \
  system.CUDA_VISIBLE_DEVICES="'0'" \
  trainer.n_gpus_per_node=1 \
  trainer.total_training_steps="${STEPS}" \
  trainer.save_freq=-1 \
  trainer.test_freq=1 \
  trainer.resume_mode=disable \
  trainer.project_name=ufb_test \
  trainer.experiment_name="${EXPERIMENT}" \
  trainer.default_local_dir="${CKPT_DIR}" \
  model_path="${MODEL_PATH}" \
  micro_batch_size_per_gpu=1 \
  ppo_mini_batch_size=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.response_length=1024 \
  actor_rollout_ref.rollout.max_model_len=8192 \
  actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
  es_manager.train.env_groups=4 \
  es_manager.train.group_size=4 \
  "es_manager.train.env_configs.tags=[${ENV_TAG}]" \
  "es_manager.train.env_configs.n_groups=[4]" \
  es_manager.val.env_groups=4 \
  es_manager.val.group_size=1 \
  "es_manager.val.env_configs.tags=[${ENV_TAG}]" \
  "es_manager.val.env_configs.n_groups=[4]" \
  2>&1 | tee "${LOG_FILE}"
