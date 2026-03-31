#!/bin/bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load cuda/12.8 || true
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate ragen
fi

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/outputs/slurm"

export PYTHONPATH="${PROJECT_DIR}/verl:${PROJECT_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=ERROR
export NCCL_P2P_DISABLE=1
export VLLM_LOG_LEVEL=WARNING
export TRANSFORMERS_VERBOSITY=warning
export HF_HUB_VERBOSITY=error
export CUDA_LAUNCH_BLOCKING=0
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export HF_TOKEN="${HF_TOKEN:-}"

BENCHMARK_NAME="${BENCHMARK_NAME:?BENCHMARK_NAME is required}"
ENV_TAG="${ENV_TAG:?ENV_TAG is required}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
FEEDBACK_MODE="${FEEDBACK_MODE:?FEEDBACK_MODE is required}"
ATTEMPT_TIMES="${ATTEMPT_TIMES:?ATTEMPT_TIMES is required}"
TURN_PER_ATTEMPT="${TURN_PER_ATTEMPT:?TURN_PER_ATTEMPT is required}"

STEPS="${STEPS:-200}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-10}"
NGPUS="${NGPUS:-1}"
TRAIN_ENV_GROUPS="${TRAIN_ENV_GROUPS:-8}"
TRAIN_GROUP_SIZE="${TRAIN_GROUP_SIZE:-16}"
VAL_ENV_GROUPS="${VAL_ENV_GROUPS:-512}"
VAL_GROUP_SIZE="${VAL_GROUP_SIZE:-1}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
CONTEXT_WINDOW_MODE="${CONTEXT_WINDOW_MODE:-full}"
MAX_CONTEXT_WINDOW="${MAX_CONTEXT_WINDOW:--1}"
ADV_ESTIMATOR="${ADV_ESTIMATOR:-gae}"
GRPO_ADVANTAGE_LENGTH_WEIGHT="${GRPO_ADVANTAGE_LENGTH_WEIGHT:-False}"
MAX_RETRY_ATTEMPTS="${MAX_RETRY_ATTEMPTS:-${ATTEMPT_TIMES}}"
MAX_TURNS_PER_ATTEMPT="${MAX_TURNS_PER_ATTEMPT:-${TURN_PER_ATTEMPT}}"
MAX_ACTIONS_PER_ATTEMPT="${MAX_ACTIONS_PER_ATTEMPT:-${MAX_TURNS_PER_ATTEMPT}}"
DEFAULT_MAX_TURN="$((MAX_RETRY_ATTEMPTS * MAX_TURNS_PER_ATTEMPT))"
MAX_TURN="${MAX_TURN:-${DEFAULT_MAX_TURN}}"
MAX_ACTIONS_PER_TRAJ="${MAX_ACTIONS_PER_TRAJ:-$((MAX_RETRY_ATTEMPTS * MAX_ACTIONS_PER_ATTEMPT))}"
REWARD_DECAY_BASE="${REWARD_DECAY_BASE:-2.0}"
VAL_MAX_TURN="${VAL_MAX_TURN:-${MAX_TURN}}"
EXPERIMENT="${EXPERIMENT:?EXPERIMENT is required}"
CKPT_DIR="${CKPT_DIR:-/workspace/ufb_exp/${EXPERIMENT}}"
DEFAULT_CUDA_DEVICES="$(seq -s, 0 $((NGPUS - 1)))"
CUDA_DEVICES="${CUDA_DEVICES:-${DEFAULT_CUDA_DEVICES}}"

ENABLE_RETRY_WRAPPER="true"
case "${ENV_TAG}" in
  MetamathQA|MetamathQANoFeedback|MetamathQASpecificFeedback)
    ENABLE_RETRY_WRAPPER="false"
    ;;
esac

RETRY_OVERRIDES=()
if [ "${ENABLE_RETRY_WRAPPER}" = "true" ]; then
  RETRY_OVERRIDES+=("custom_envs.${ENV_TAG}.max_actions_per_traj=${MAX_ACTIONS_PER_TRAJ}")
  RETRY_OVERRIDES+=("++custom_envs.${ENV_TAG}.retry.max_turns_per_attempt=${MAX_TURNS_PER_ATTEMPT}")
  RETRY_OVERRIDES+=("++custom_envs.${ENV_TAG}.retry.max_actions_per_attempt=${MAX_ACTIONS_PER_ATTEMPT}")
  RETRY_OVERRIDES+=("++custom_envs.${ENV_TAG}.retry.max_retry_attempts=${MAX_RETRY_ATTEMPTS}")
  RETRY_OVERRIDES+=("++custom_envs.${ENV_TAG}.retry.reward_decay_base=${REWARD_DECAY_BASE}")
fi

mkdir -p "${CKPT_DIR}"

echo "[INFO] ============================================"
echo "[INFO] Benchmark:   ${BENCHMARK_NAME}"
echo "[INFO] Env tag:     ${ENV_TAG}"
echo "[INFO] Feedback:    ${FEEDBACK_MODE}"
echo "[INFO] Attempts:    ${ATTEMPT_TIMES}"
echo "[INFO] Turns/att:   ${TURN_PER_ATTEMPT}"
echo "[INFO] Max turn:    ${MAX_TURN}"
echo "[INFO] Model:       ${MODEL_PATH}"
echo "[INFO] GPUs:        ${NGPUS} (${CUDA_DEVICES})"
echo "[INFO] Steps:       ${STEPS}"
echo "[INFO] Train grp:   ${TRAIN_ENV_GROUPS} x ${TRAIN_GROUP_SIZE}"
echo "[INFO] Val grp:     ${VAL_ENV_GROUPS} x ${VAL_GROUP_SIZE}"
echo "[INFO] Ctx mode:    ${CONTEXT_WINDOW_MODE}"
echo "[INFO] Ctx window:  ${MAX_CONTEXT_WINDOW}"
echo "[INFO] Adv est:     ${ADV_ESTIMATOR}"
if [ "${ADV_ESTIMATOR}" = "grpo" ]; then
  echo "[INFO] GRPO len wt: ${GRPO_ADVANTAGE_LENGTH_WEIGHT}"
fi
echo "[INFO] Retry wrap:  ${ENABLE_RETRY_WRAPPER}"
if [ "${ENABLE_RETRY_WRAPPER}" = "true" ]; then
  echo "[INFO] Retry turns: ${MAX_TURNS_PER_ATTEMPT}"
  echo "[INFO] Retry acts:  ${MAX_ACTIONS_PER_ATTEMPT}"
  echo "[INFO] Max acts/r:  ${MAX_ACTIONS_PER_TRAJ}"
  echo "[INFO] Decay base:  ${REWARD_DECAY_BASE}"
fi
echo "[INFO] Checkpoint:  ${CKPT_DIR}"
echo "[INFO] Note:        MetaMathQA tags skip wrapper retry; others may use attempt-level retry."
echo "[INFO] ============================================"

python train.py \
  --config-name="base" \
  system.CUDA_VISIBLE_DEVICES="'${CUDA_DEVICES}'" \
  trainer.n_gpus_per_node="${NGPUS}" \
  trainer.total_training_steps="${STEPS}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.validation_steps=1 \
  trainer.resume_mode=disable \
  trainer.project_name=ufb_train_batch \
  trainer.experiment_name="${EXPERIMENT}" \
  trainer.default_local_dir="${CKPT_DIR}" \
  model_path="${MODEL_PATH}" \
  algorithm.adv_estimator="${ADV_ESTIMATOR}" \
  grpo_advantage_length_weight="${GRPO_ADVANTAGE_LENGTH_WEIGHT}" \
  ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.entropy_coeff=0.001 \
  agent_proxy.max_turn="${MAX_TURN}" \
  agent_proxy.context_window_mode="${CONTEXT_WINDOW_MODE}" \
  agent_proxy.max_context_window="${MAX_CONTEXT_WINDOW}" \
  val_agent_proxy.max_turn="${VAL_MAX_TURN}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
  es_manager.train.env_groups="${TRAIN_ENV_GROUPS}" \
  es_manager.train.group_size="${TRAIN_GROUP_SIZE}" \
  es_manager.train.env_configs.tags="['${ENV_TAG}']" \
  es_manager.train.env_configs.n_groups="[${TRAIN_ENV_GROUPS}]" \
  es_manager.val.env_groups="${VAL_ENV_GROUPS}" \
  es_manager.val.group_size="${VAL_GROUP_SIZE}" \
  es_manager.val.env_configs.tags="['${ENV_TAG}']" \
  es_manager.val.env_configs.n_groups="[${VAL_ENV_GROUPS}]" \
  "${RETRY_OVERRIDES[@]}" \
  +trainer.max_actor_ckpt_to_keep=4 \
  +trainer.max_critic_ckpt_to_keep=0 \
  +actor_rollout_ref.actor.checkpoint.contents=[model,optimizer,extra,hf_config]
