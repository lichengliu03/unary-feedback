#!/bin/bash
#SBATCH -J eval_independent_passk
#SBATCH -N 1
#SBATCH --partition=gpuH200x8
#SBATCH --account=bfea-delta-gpu
#SBATCH --gres=gpu:h200:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o /u/lliu22/unary-feedback/slurm-eval_independent_passk-%j.out
#SBATCH -e /u/lliu22/unary-feedback/slurm-eval_independent_passk-%j.err

set -euo pipefail

module purge
module load cuda/12.8

if [[ -f /u/lliu22/miniconda3/etc/profile.d/conda.sh ]]; then
  source /u/lliu22/miniconda3/etc/profile.d/conda.sh
else
  source /u/lliu22/miniconda3/bin/activate
fi
conda activate ragen

cd /u/lliu22/unary-feedback

export PYTHONPATH="/u/lliu22/unary-feedback:${PYTHONPATH:-}"
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

echo "[INFO] Launching Independent Pass@k evaluation on ${SLURM_NODELIST}"

# ====== Configuration ======
# Model to evaluate - can be set via environment variable or default to base
export MODEL_NAME="${MODEL_NAME:-qwen25_3b_5turn_200steps}"

# Only set CHECKPOINT_STEP default if not base_model
if [[ "${MODEL_NAME}" == "base_model" ]]; then
  export CHECKPOINT_STEP=""
  export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
else
  export CHECKPOINT_STEP="${CHECKPOINT_STEP:-global_step_200}"
fi

# Evaluation parameters
export DEVICES="0,1"
export NUM_SAMPLES="${NUM_SAMPLES:-512}"
export K_VALUES="${K_VALUES:-1,2,4,8,16,32,64,128,256,512}"
export EVAL_QUESTIONS="${EVAL_QUESTIONS:-30}"
export TEMPERATURE="${TEMPERATURE:-0.8}"
export TOP_P="${TOP_P:-0.95}"
export DATASET_TAG="AIME24"

echo "[INFO] Model: ${MODEL_NAME}"
echo "[INFO] Checkpoint: ${CHECKPOINT_STEP}"
echo "[INFO] Dataset: ${DATASET_TAG}"
echo "[INFO] Samples per problem: ${NUM_SAMPLES}"
echo "[INFO] K values: ${K_VALUES}"
echo "[INFO] Questions: ${EVAL_QUESTIONS}"
echo "[INFO] Temperature: ${TEMPERATURE}, Top-p: ${TOP_P}"

# Run evaluation script
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  bash scripts/evaluation/eval_independent_passk.sh

echo "[DONE] Evaluation complete!"
