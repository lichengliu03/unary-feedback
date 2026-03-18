#!/bin/bash
#SBATCH -J eval_converted_ckpts
#SBATCH -N 1
#SBATCH --partition=gpuH200x8
#SBATCH --account=bfea-delta-gpu
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH -t 08:00:00
#SBATCH -o outputs/slurm/slurm-%x-%j.out
#SBATCH -e outputs/slurm/slurm-%x-%j.err

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

echo "[INFO] Launching eval on ${SLURM_NODELIST}"
echo "[INFO] Evaluating: ${EXP_NAME}"

DEVICES="0,1" \
CKPT_DIR="/projects/bfea/lliu22/ragen_checkpoints/${EXP_NAME}" \
CKPT_STEPS="50 100 150 200" \
SKIP_BASE_MODEL=1 \
EVAL_MAX_TURN=5 \
EVAL_QUESTIONS=1024 \
EVAL_BATCH_SIZE=512 \
EVAL_K_VALUES="1 2 3 4 5" \
EVAL_GPU_MEMORY_UTIL=0.5 \
EVAL_RESPONSE_LENGTH=400 \
EVAL_MAX_MODEL_LEN=16384 \
ROLLOUT_LOAD_FORMAT="safetensors" \
ROLLOUT_LAYERED_SUMMON=1 \
WANDB_PROJECT=ufb_metamathqa_eval \
SKIP_BASHRC=1 \
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  bash scripts/evaluation/eval_checkpoints_normal_feedback.sh
