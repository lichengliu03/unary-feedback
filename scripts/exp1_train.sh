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
#   bash scripts/exp1_train.sh                                  # default: MetamathQA, 2 GPUs
#   NGPUS=1 bash scripts/exp1_train.sh                          # 1 GPU
#   ENV_TAG=SimpleSokoban NGPUS=1 bash scripts/exp1_train.sh
#   ENV_TAG=FrozenLake NGPUS=2 bash scripts/exp1_train.sh
#   RETRY_MODE=no_retry bash scripts/exp1_train.sh              # disable retry
#   ENV_TAG=FrozenLake RETRY_MODE=no_retry bash scripts/exp1_train.sh
#
# Available ENV_TAGs (from configs/envs.yaml):
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
ENV_TAG="${ENV_TAG:-MetamathQA}"
RETRY_MODE="${RETRY_MODE:-retry}"   # retry | no_retry
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
STEPS="${STEPS:-200}"
NGPUS="${NGPUS:-2}"
SAVE_FREQ="${SAVE_FREQ:-50}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-512}"
MAX_TURNS_PER_ATTEMPT="${MAX_TURNS_PER_ATTEMPT:-5}"
MAX_ACTIONS_PER_ATTEMPT="${MAX_ACTIONS_PER_ATTEMPT:-10}"

case "${RETRY_MODE}" in
  retry|no_retry)
    ;;
  *)
    echo "[ERROR] Unsupported RETRY_MODE: ${RETRY_MODE}" >&2
    echo "[ERROR] Expected one of: retry, no_retry" >&2
    exit 1
    ;;
esac

ENV_USES_RETRY_WRAPPER=false
case "${ENV_TAG}" in
  SimpleSokoban|LargerSokoban|FrozenLake)
    ENV_USES_RETRY_WRAPPER=true
    ;;
esac

# Retry semantics:
#   - MetamathQA / Countdown: each turn is one attempt
#   - Sokoban / FrozenLake: attempts come from MultiTurnRetryWrapper
if [ "${ENV_USES_RETRY_WRAPPER}" = "true" ]; then
  if [ "${RETRY_MODE}" = "no_retry" ]; then
    DEFAULT_MAX_RETRY_ATTEMPTS=1
  else
    DEFAULT_MAX_RETRY_ATTEMPTS=3
  fi
  MAX_RETRY_ATTEMPTS="${MAX_RETRY_ATTEMPTS:-${DEFAULT_MAX_RETRY_ATTEMPTS}}"
  DEFAULT_MAX_TURN="$((MAX_RETRY_ATTEMPTS * MAX_TURNS_PER_ATTEMPT))"
else
  if [ "${RETRY_MODE}" = "no_retry" ]; then
    DEFAULT_MAX_TURN=1
  else
    DEFAULT_MAX_TURN=5
  fi
fi

MAX_TURN="${MAX_TURN:-${DEFAULT_MAX_TURN}}"

DEFAULT_EXPERIMENT="exp1_${ENV_TAG}"
if [ "${RETRY_MODE}" = "no_retry" ]; then
  DEFAULT_EXPERIMENT="${DEFAULT_EXPERIMENT}_no_retry"
fi
EXPERIMENT="${EXPERIMENT:-${DEFAULT_EXPERIMENT}}"
CKPT_DIR="${CKPT_DIR:-${PROJECT_DIR}/outputs/checkpoints/${EXPERIMENT}}"
LOG_FILE="${LOG_FILE:-${PROJECT_DIR}/outputs/logs/${EXPERIMENT}.log}"

# Build CUDA device list: "0" for 1 GPU, "0,1" for 2 GPUs
CUDA_DEVICES=$(seq -s, 0 $((NGPUS - 1)))

RETRY_OVERRIDES=()
if [ "${ENV_USES_RETRY_WRAPPER}" = "true" ]; then
  RETRY_OVERRIDES+=("custom_envs.${ENV_TAG}.max_actions_per_traj=${MAX_ACTIONS_PER_ATTEMPT}")
  RETRY_OVERRIDES+=("custom_envs.${ENV_TAG}.retry.max_turns_per_attempt=${MAX_TURNS_PER_ATTEMPT}")
  RETRY_OVERRIDES+=("custom_envs.${ENV_TAG}.retry.max_actions_per_attempt=${MAX_ACTIONS_PER_ATTEMPT}")
  RETRY_OVERRIDES+=("custom_envs.${ENV_TAG}.retry.max_retry_attempts=${MAX_RETRY_ATTEMPTS}")
fi

mkdir -p "${CKPT_DIR}"

echo "[INFO] ============================================"
echo "[INFO] Training"
echo "[INFO] Env tag:     ${ENV_TAG}"
echo "[INFO] Retry mode:  ${RETRY_MODE}"
echo "[INFO] GPUs:        ${NGPUS} (${CUDA_DEVICES})"
echo "[INFO] Model:       ${MODEL_PATH}"
echo "[INFO] Steps:       ${STEPS}"
echo "[INFO] Max turn:    ${MAX_TURN}"
if [ "${ENV_USES_RETRY_WRAPPER}" = "true" ]; then
  echo "[INFO] Attempts:    ${MAX_RETRY_ATTEMPTS}"
  echo "[INFO] Turns/att:   ${MAX_TURNS_PER_ATTEMPT}"
  echo "[INFO] Acts/att:    ${MAX_ACTIONS_PER_ATTEMPT}"
else
  echo "[INFO] Attempts:    ${MAX_TURN} (1 turn/attempt)"
fi
echo "[INFO] Resp len:    ${RESPONSE_LENGTH}"
echo "[INFO] Checkpoint:  ${CKPT_DIR}"
echo "[INFO] Log file:    ${LOG_FILE}"
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
  trainer.experiment_name="${EXPERIMENT}" \
  trainer.default_local_dir="${CKPT_DIR}" \
  model_path="${MODEL_PATH}" \
  ppo_micro_batch_size_per_gpu=2 \
  log_prob_micro_batch_size_per_gpu=8 \
  ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  agent_proxy.max_turn="${MAX_TURN}" \
  val_agent_proxy.max_turn="${MAX_TURN}" \
  actor_rollout_ref.rollout.response_length="${RESPONSE_LENGTH}" \
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
  "${RETRY_OVERRIDES[@]}" \
  +trainer.max_actor_ckpt_to_keep=4 \
  +trainer.max_critic_ckpt_to_keep=0 \
  "+actor_rollout_ref.actor.checkpoint.contents=[model,optimizer,extra,hf_config]" \
  2>&1 | tee "${LOG_FILE}"
