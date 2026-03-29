#!/bin/bash
#SBATCH --partition=gpuH200x8
#SBATCH --account=bflz-delta-gpu
#SBATCH -J webshop_q25_3b
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH -t 24:00:00
#SBATCH -o /u/ylin30/unary-feedback/outputs/slurm/slurm-%x-%j.out
#SBATCH -e /u/ylin30/unary-feedback/outputs/slurm/slurm-%x-%j.err

set -euo pipefail

export ENV_TAG="WebShop"
export MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"
export MAX_RETRY_ATTEMPTS="2"
export MAX_TURNS_PER_ATTEMPT="7"
export MAX_ACTIONS_PER_TURN="1"
export MAX_ACTIONS_PER_TRAJ="7"
export MAX_ACTIONS_PER_ATTEMPT="7"
export MAX_TURN="14"
export EXPERIMENT="webshop_qwen25_3b_one_bit_2attempts_7turns"

resolve_webshop_runner() {
  local runner=""
  local command_path=""

  if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
    command_path="$(scontrol show job "${SLURM_JOB_ID}" -o | sed -n 's/.* Command=\([^ ]*\).*/\1/p')"
    if [ -n "${command_path}" ]; then
      runner="$(cd "$(dirname "${command_path}")/../webshop" && pwd)/train_qwen25_3b.sh"
    fi
  fi

  if [ -z "${runner}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    if [ -f "${SLURM_SUBMIT_DIR}/scripts/webshop/train_qwen25_3b.sh" ]; then
      runner="${SLURM_SUBMIT_DIR}/scripts/webshop/train_qwen25_3b.sh"
    elif [ -f "${SLURM_SUBMIT_DIR}/train_qwen25_3b.sh" ]; then
      runner="${SLURM_SUBMIT_DIR}/train_qwen25_3b.sh"
    fi
  fi

  if [ -z "${runner}" ]; then
    runner="$(cd "$(dirname "${BASH_SOURCE[0]}")/../webshop" && pwd)/train_qwen25_3b.sh"
  fi

  printf '%s\n' "${runner}"
}

WEBSHOP_RUNNER="$(resolve_webshop_runner)"
if [ ! -f "${WEBSHOP_RUNNER}" ]; then
  echo "[ERROR] Could not locate train_qwen25_3b.sh from job wrapper" >&2
  exit 1
fi

bash "${WEBSHOP_RUNNER}"
