#!/bin/bash
#SBATCH -J ufb_eval
#SBATCH -N 1
#SBATCH --partition=gpuH200x8
#SBATCH --account=bfea-delta-gpu
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH -t 04:00:00
#SBATCH -o /u/lliu22/unary-feedback/outputs/slurm/slurm-%x-%j.out
#SBATCH -e /u/lliu22/unary-feedback/outputs/slurm/slurm-%x-%j.err

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

export PYTHONPATH="/u/lliu22/unary-feedback/verl:/u/lliu22/unary-feedback:${PYTHONPATH:-}"
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

# ============================================================
# Usage:
#   ENV=metamathqa MODEL=Qwen/Qwen2.5-3B-Instruct sbatch scripts/evaluation/submit_eval.sh
#   ENV=sokoban MODEL=/path/to/checkpoint sbatch scripts/evaluation/submit_eval.sh
#
# Environment variables:
#   ENV           - Environment name (REQUIRED, matches configs/envs/<ENV>.yaml)
#   MODEL         - Model path or checkpoint path (REQUIRED)
#   VAL_GROUPS    - Number of validation groups (default: 256)
#   EVAL_TURN     - Max turns during evaluation (default: from env yaml)
# ============================================================

ENV="${ENV:?ERROR: ENV is required. Example: ENV=metamathqa}"
MODEL="${MODEL:?ERROR: MODEL is required. Example: MODEL=Qwen/Qwen2.5-3B-Instruct}"
VAL_GROUPS="${VAL_GROUPS:-256}"
EVAL_TURN="${EVAL_TURN:-}"  # empty = use env yaml default

echo "[INFO] ============================================"
echo "[INFO] Evaluation"
echo "[INFO] Environment: ${ENV}"
echo "[INFO] Model:       ${MODEL}"
echo "[INFO] Val groups:  ${VAL_GROUPS}"
echo "[INFO] Eval turn:   ${EVAL_TURN:-<from env yaml>}"
echo "[INFO] ============================================"

# Build optional turn override
TURN_OVERRIDE=""
if [[ -n "${EVAL_TURN}" ]]; then
  TURN_OVERRIDE="val_agent_proxy.max_turn=${EVAL_TURN}"
fi

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  python train.py \
    --config-name="envs/${ENV}" \
    system.CUDA_VISIBLE_DEVICES="'0,1'" \
    trainer.n_gpus_per_node=2 \
    trainer.total_training_steps=0 \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.project_name=ufb_eval \
    trainer.experiment_name="eval_${ENV}_$(basename ${MODEL})" \
    model_path="${MODEL}" \
    micro_batch_size_per_gpu=2 \
    ppo_mini_batch_size=16 \
    es_manager.val.env_groups="${VAL_GROUPS}" \
    es_manager.val.env_configs.n_groups="[${VAL_GROUPS}]" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    ${TURN_OVERRIDE}
