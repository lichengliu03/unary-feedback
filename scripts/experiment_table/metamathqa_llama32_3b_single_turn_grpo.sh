#!/bin/bash
#SBATCH --partition=gpuH200x8
#SBATCH --account=bfea-delta-gpu
#SBATCH -J meta_ll32_stg
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH -t 06:00:00

set -euo pipefail

export BENCHMARK_NAME="MetaMathQA"
export ENV_TAG="MetamathQA"
export MODEL_PATH="meta-llama/Llama-3.2-3B-Instruct"
export FEEDBACK_MODE="one-bit"
export NGPUS="2"
export ADV_ESTIMATOR="grpo"
export ATTEMPT_TIMES="1"
export TURN_PER_ATTEMPT="1"
export MAX_TURN="1"
export EXPERIMENT="metamathqa_llama32_3b_single_turn_grpo_1attempt_1turn"

# When launched via sbatch, this wrapper runs from Slurm's spool dir.
resolve_experiment_runner() {
  local helper=""
  local command_path=""

  if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
    command_path="$(scontrol show job "${SLURM_JOB_ID}" -o | sed -n 's/.* Command=\([^ ]*\).*/\1/p')"
    if [ -n "${command_path}" ]; then
      helper="$(cd "$(dirname "${command_path}")" && pwd)/_train_standard_experiment.sh"
    fi
  fi

  if [ -z "${helper}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    if [ -f "${SLURM_SUBMIT_DIR}/_train_standard_experiment.sh" ]; then
      helper="${SLURM_SUBMIT_DIR}/_train_standard_experiment.sh"
    elif [ -f "${SLURM_SUBMIT_DIR}/scripts/experiment_table/_train_standard_experiment.sh" ]; then
      helper="${SLURM_SUBMIT_DIR}/scripts/experiment_table/_train_standard_experiment.sh"
    fi
  fi

  if [ -z "${helper}" ]; then
    helper="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_train_standard_experiment.sh"
  fi

  printf '%s\n' "${helper}"
}

HELPER_SCRIPT="$(resolve_experiment_runner)"
if [ ! -f "${HELPER_SCRIPT}" ]; then
  echo "[ERROR] Could not locate _train_standard_experiment.sh from job wrapper" >&2
  exit 1
fi

bash "${HELPER_SCRIPT}"
