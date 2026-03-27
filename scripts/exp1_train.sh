#!/bin/bash
# EXP1: Training for any environment with 1 or 2 GPUs.
#
# Tested environments:
#   1. MetamathQA
#   2. Countdown
#   3. SimpleSokoban
#   4. FrozenLake
#
# Usage:
#   bash scripts/exp1_train.sh                                  # default: loop over 4 envs
#   NGPUS=1 bash scripts/exp1_train.sh                          # 1 GPU
#   ENV_TAG=SimpleSokoban NGPUS=1 bash scripts/exp1_train.sh
#   ENV_TAGS=MetamathQA,Countdown NGPUS=1 bash scripts/exp1_train.sh
#
# Available ENV_TAG / ENV_TAGS values (from configs/envs.yaml):
#   MetamathQA, Countdown, SimpleSokoban, FrozenLake, Bandit,
#   HotpotQA, GSM8k, MATH, AIME24, WebShop, SimpleSudoku, ...

set -euo pipefail

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
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
STEPS="${STEPS:-200}"
NGPUS="${NGPUS:-1}"
SAVE_FREQ="${SAVE_FREQ:-50}"
DEFAULT_ENV_TAGS=("MetamathQA" "Countdown" "SimpleSokoban" "FrozenLake")

ENV_TAG_ARRAY=()
if [ -n "${ENV_TAG:-}" ]; then
  ENV_TAG_ARRAY=("${ENV_TAG}")
elif [ -n "${ENV_TAGS:-}" ]; then
  IFS=',' read -r -a RAW_ENV_TAG_ARRAY <<< "${ENV_TAGS}"
  for raw_tag in "${RAW_ENV_TAG_ARRAY[@]}"; do
    tag="${raw_tag//[[:space:]]/}"
    if [ -n "${tag}" ]; then
      ENV_TAG_ARRAY+=("${tag}")
    fi
  done
else
  ENV_TAG_ARRAY=("${DEFAULT_ENV_TAGS[@]}")
fi

if [ "${#ENV_TAG_ARRAY[@]}" -eq 0 ]; then
  echo "[ERROR] No environment tags were provided." >&2
  exit 1
fi

# Build CUDA device list: "0" for 1 GPU, "0,1" for 2 GPUs
CUDA_DEVICES=$(seq -s, 0 $((NGPUS - 1)))

echo "[INFO] Selected env tags: ${ENV_TAG_ARRAY[*]}"

for ENV_TAG in "${ENV_TAG_ARRAY[@]}"; do
  # Set max_turn based on environment type:
  #   Single-turn (MetamathQA, Countdown): max_turn = max retries (5)
  #   Multi-turn (SimpleSokoban, FrozenLake): max_turn = turns_per_attempt * retries (15)
  if [ -n "${MAX_TURN:-}" ]; then
    RUN_MAX_TURN="${MAX_TURN}"
  else
    case "${ENV_TAG}" in
      SimpleSokoban|LargerSokoban|FrozenLake)
        RUN_MAX_TURN=15
        ;;
      *)
        RUN_MAX_TURN=5
        ;;
    esac
  fi

  if [ "${#ENV_TAG_ARRAY[@]}" -eq 1 ]; then
    RUN_EXPERIMENT="${EXPERIMENT:-exp1_${ENV_TAG}}"
    RUN_CKPT_DIR="${CKPT_DIR:-${PROJECT_DIR}/outputs/checkpoints/${RUN_EXPERIMENT}}"
    RUN_LOG_FILE="${LOG_FILE:-${PROJECT_DIR}/outputs/logs/exp1_${ENV_TAG}.log}"
  else
    RUN_EXPERIMENT="exp1_${ENV_TAG}"
    RUN_CKPT_DIR="${PROJECT_DIR}/outputs/checkpoints/${RUN_EXPERIMENT}"
    RUN_LOG_FILE="${PROJECT_DIR}/outputs/logs/exp1_${ENV_TAG}.log"
  fi

  mkdir -p "${RUN_CKPT_DIR}"

  echo "[INFO] ============================================"
  echo "[INFO] Training"
  echo "[INFO] Env tag:     ${ENV_TAG}"
  echo "[INFO] GPUs:        ${NGPUS} (${CUDA_DEVICES})"
  echo "[INFO] Model:       ${MODEL_PATH}"
  echo "[INFO] Steps:       ${STEPS}"
  echo "[INFO] Max turn:    ${RUN_MAX_TURN}"
  echo "[INFO] Checkpoint:  ${RUN_CKPT_DIR}"
  echo "[INFO] Log file:    ${RUN_LOG_FILE}"
  echo "[INFO] ============================================"

  # Use base config directly to avoid Hydra defaults chain issues with envs/*.yaml
  python train.py \
    --config-name=base \
    system.CUDA_VISIBLE_DEVICES="'${CUDA_DEVICES}'" \
    trainer.n_gpus_per_node="${NGPUS}" \
    trainer.total_training_steps="${STEPS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq=10 \
    trainer.resume_mode=disable \
    trainer.project_name=ufb_train \
    trainer.experiment_name="${RUN_EXPERIMENT}" \
    trainer.default_local_dir="${RUN_CKPT_DIR}" \
    model_path="${MODEL_PATH}" \
    ppo_micro_batch_size_per_gpu=2 \
    log_prob_micro_batch_size_per_gpu=4 \
    ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    agent_proxy.max_turn="${RUN_MAX_TURN}" \
    val_agent_proxy.max_turn="${RUN_MAX_TURN}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    es_manager.train.env_groups=8 \
    es_manager.train.group_size=16 \
    "es_manager.train.env_configs.tags=[${ENV_TAG}]" \
    "es_manager.train.env_configs.n_groups=[8]" \
    es_manager.val.env_groups=512 \
    es_manager.val.group_size=1 \
    "es_manager.val.env_configs.tags=[${ENV_TAG}]" \
    "es_manager.val.env_configs.n_groups=[512]" \
    +trainer.max_actor_ckpt_to_keep=4 \
    +trainer.max_critic_ckpt_to_keep=0 \
    "+actor_rollout_ref.actor.checkpoint.contents=[model,optimizer,extra,hf_config]" \
    2>&1 | tee "${RUN_LOG_FILE}"
done
