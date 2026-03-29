#!/bin/bash
#SBATCH --partition=gpuH200x8
#SBATCH --account=bflz-delta-gpu
#SBATCH -J webshop_retry_rl
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH -t 24:00:00
#SBATCH -o /u/ylin30/unary-feedback/outputs/slurm/slurm-%x-%j.out
#SBATCH -e /u/ylin30/unary-feedback/outputs/slurm/slurm-%x-%j.err

set -euo pipefail

resolve_project_dir() {
  local script_path=""

  if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
    script_path="$(scontrol show job "${SLURM_JOB_ID}" -o | sed -n 's/.* Command=\([^ ]*\).*/\1/p')"
  fi

  if [ -z "${script_path}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    if [ -f "${SLURM_SUBMIT_DIR}/scripts/webshop/train_qwen25_3b.sh" ]; then
      script_path="${SLURM_SUBMIT_DIR}/scripts/webshop/train_qwen25_3b.sh"
    elif [ -f "${SLURM_SUBMIT_DIR}/train_qwen25_3b.sh" ]; then
      script_path="${SLURM_SUBMIT_DIR}/train_qwen25_3b.sh"
    fi
  fi

  if [ -z "${script_path}" ]; then
    script_path="${BASH_SOURCE[0]}"
  fi

  cd "$(dirname "${script_path}")/../.." && pwd
}

PROJECT_DIR="$(resolve_project_dir)"

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load cuda/12.8 || true
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate ragen
fi

find_java21_home() {
  local candidates=()
  local home=""
  local candidate=""

  if [ -n "${JAVA_HOME:-}" ]; then
    candidates+=("${JAVA_HOME}")
  fi
  if [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/lib/jvm" ]; then
    candidates+=("${CONDA_PREFIX}/lib/jvm")
  fi
  for candidate in /u/ylin30/.conda/pkgs/openjdk-21*/lib/jvm /usr/lib/jvm/java-21* /usr/lib/jvm/jdk-21*; do
    if [ -d "${candidate}" ]; then
      candidates+=("${candidate}")
    fi
  done

  for home in "${candidates[@]}"; do
    if [ -x "${home}/bin/java" ] && "${home}/bin/java" -version 2>&1 | grep -q 'version "21'; then
      printf '%s\n' "${home}"
      return 0
    fi
  done
  return 1
}

JAVA21_HOME="$(find_java21_home || true)"
if [ -z "${JAVA21_HOME}" ]; then
  echo "[ERROR] WebShop requires Java 21 for pyserini/anserini."
  echo "[ERROR] Set JAVA_HOME to a JDK 21 install and rerun."
  exit 1
fi

export JAVA_HOME="${JAVA21_HOME}"
export JVM_PATH="${JAVA_HOME}/lib/server/libjvm.so"
export PATH="${JAVA_HOME}/bin:${PATH}"

if ! python - <<'PY' >/dev/null 2>&1
import spacy
spacy.load("en_core_web_sm")
PY
then
  echo "[ERROR] Missing spaCy model en_core_web_sm."
  echo "[ERROR] Install it with: python -m spacy download en_core_web_sm"
  exit 1
fi

cd "${PROJECT_DIR}"

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
export HF_TOKEN="${HF_TOKEN:-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}"
export PYTHONPATH="${PROJECT_DIR}/external/webshop-minimal:${PYTHONPATH:-}"

ENV_TAG="${ENV_TAG:-WebShop}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
NGPUS="${NGPUS:-2}"
STEPS="${STEPS:-200}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-10}"
TRAIN_ENV_GROUPS="${TRAIN_ENV_GROUPS:-8}"
TRAIN_GROUP_SIZE="${TRAIN_GROUP_SIZE:-16}"
VAL_ENV_GROUPS="${VAL_ENV_GROUPS:-512}"
VAL_GROUP_SIZE="${VAL_GROUP_SIZE:-1}"
MAX_ACTIONS_PER_TURN="${MAX_ACTIONS_PER_TURN:-1}"
MAX_ACTIONS_PER_TRAJ="${MAX_ACTIONS_PER_TRAJ:-8}"
CONTEXT_WINDOW_MODE="${CONTEXT_WINDOW_MODE:-full}"
MAX_CONTEXT_WINDOW="${MAX_CONTEXT_WINDOW:--1}"
MAX_RETRY_ATTEMPTS="${MAX_RETRY_ATTEMPTS:-${MAX_ATTEMPTS:-2}}"
MAX_TURNS_PER_ATTEMPT="${MAX_TURNS_PER_ATTEMPT:-8}"
MAX_ACTIONS_PER_ATTEMPT="${MAX_ACTIONS_PER_ATTEMPT:-${MAX_ACTIONS_PER_TRAJ}}"
DEFAULT_MAX_TURN="$((MAX_TURNS_PER_ATTEMPT * MAX_RETRY_ATTEMPTS))"
MAX_TURN="${MAX_TURN:-${DEFAULT_MAX_TURN}}"
VAL_MAX_TURN="${VAL_MAX_TURN:-${MAX_TURN}}"
RANDOMIZE_FEEDBACK="${RANDOMIZE_FEEDBACK:-true}"
REWARD_DECAY_BASE="${REWARD_DECAY_BASE:-2.0}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8100}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
VAL_INSTANCES="$((VAL_ENV_GROUPS * VAL_GROUP_SIZE))"
EXPERIMENT="${EXPERIMENT:-qwen25_3b_webshop_retry_${STEPS}steps_${MAX_RETRY_ATTEMPTS}attempts_${MAX_TURNS_PER_ATTEMPT}turns_val${VAL_INSTANCES}}"
CKPT_DIR="${CKPT_DIR:-/projects/bflz/${EXPERIMENT}}"
DEFAULT_CUDA_DEVICES="$(seq -s, 0 $((NGPUS - 1)))"
CUDA_DEVICES="${CUDA_DEVICES:-${DEFAULT_CUDA_DEVICES}}"

mkdir -p "${CKPT_DIR}"

echo "[INFO] ============================================"
echo "[INFO] Environment: ${ENV_TAG}"
echo "[INFO] Model:       ${MODEL_PATH}"
echo "[INFO] Steps:       ${STEPS}"
echo "[INFO] Save freq:   ${SAVE_FREQ}"
echo "[INFO] Train grp:   ${TRAIN_ENV_GROUPS} x ${TRAIN_GROUP_SIZE}"
echo "[INFO] Val grp:     ${VAL_ENV_GROUPS} x ${VAL_GROUP_SIZE} (${VAL_INSTANCES})"
echo "[INFO] Max turn:    ${MAX_TURN}"
echo "[INFO] Val turn:    ${VAL_MAX_TURN}"
echo "[INFO] Max acts/t:  ${MAX_ACTIONS_PER_TURN}"
echo "[INFO] Max acts/a:  ${MAX_ACTIONS_PER_TRAJ}"
echo "[INFO] Retry turns: ${MAX_TURNS_PER_ATTEMPT}"
echo "[INFO] Retry acts:  ${MAX_ACTIONS_PER_ATTEMPT}"
echo "[INFO] Ctx mode:    ${CONTEXT_WINDOW_MODE}"
echo "[INFO] Ctx window:  ${MAX_CONTEXT_WINDOW}"
echo "[INFO] Attempts:    ${MAX_RETRY_ATTEMPTS}"
echo "[INFO] Rand fb:     ${RANDOMIZE_FEEDBACK}"
echo "[INFO] PPO micro:   ${PPO_MICRO_BATCH_SIZE_PER_GPU}"
echo "[INFO] Logprob mb:  ${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
echo "[INFO] PPO mini:    ${PPO_MINI_BATCH_SIZE}"
echo "[INFO] Java home:   ${JAVA_HOME}"
echo "[INFO] Checkpoint:  ${CKPT_DIR}"
echo "[INFO] ============================================"

python train.py \
  --config-name="base" \
  system.CUDA_VISIBLE_DEVICES="'${CUDA_DEVICES}'" \
  trainer.n_gpus_per_node="${NGPUS}" \
  trainer.total_training_steps="${STEPS}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.validation_steps=1 \
  agent_proxy.max_turn="${MAX_TURN}" \
  agent_proxy.max_actions_per_turn="${MAX_ACTIONS_PER_TURN}" \
  agent_proxy.context_window_mode="${CONTEXT_WINDOW_MODE}" \
  agent_proxy.max_context_window="${MAX_CONTEXT_WINDOW}" \
  val_agent_proxy.max_turn="${VAL_MAX_TURN}" \
  trainer.resume_mode=disable \
  trainer.project_name=ufb_train \
  trainer.experiment_name="${EXPERIMENT}" \
  trainer.default_local_dir="${CKPT_DIR}" \
  model_path="${MODEL_PATH}" \
  ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
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
  custom_envs.${ENV_TAG}.max_actions_per_traj="${MAX_ACTIONS_PER_TRAJ}" \
  custom_envs.${ENV_TAG}.retry.max_turns_per_attempt="${MAX_TURNS_PER_ATTEMPT}" \
  custom_envs.${ENV_TAG}.retry.max_actions_per_attempt="${MAX_ACTIONS_PER_ATTEMPT}" \
  custom_envs.${ENV_TAG}.retry.max_retry_attempts="${MAX_RETRY_ATTEMPTS}" \
  custom_envs.${ENV_TAG}.retry.reward_decay_base="${REWARD_DECAY_BASE}" \
  +trainer.max_actor_ckpt_to_keep=4 \
  +trainer.max_critic_ckpt_to_keep=0 \
  +actor_rollout_ref.actor.checkpoint.contents=[model,optimizer,extra,hf_config] 
