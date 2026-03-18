#!/bin/bash
#SBATCH -J ufb_train
#SBATCH -N 1
#SBATCH --partition=gpuH200x8
#SBATCH --account=bfea-delta-gpu
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH -t 24:00:00
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
#   ENV=MetamathQA sbatch scripts/training/submit_train.sh
#   ENV=SimpleSokoban MODEL_PATH=Qwen/Qwen2.5-7B-Instruct sbatch scripts/training/submit_train.sh
#   ENV=Countdown STEPS=100 TURN=3 sbatch scripts/training/submit_train.sh
#
# Environment variables:
#   ENV           - Environment tag from configs/envs.yaml (REQUIRED)
#   MODEL_PATH    - Model to train (default: Qwen/Qwen2.5-3B-Instruct)
#   STEPS         - Training steps (default: 200)
#   TURN          - Max turns per episode (default: 5)
#   CONFIG        - Config name (default: base)
#   TRAIN_GROUPS  - Number of training env groups (default: 8)
#   VAL_GROUPS    - Number of validation env groups (default: 128)
# ============================================================

ENV="${ENV:?ERROR: ENV is required. Example: ENV=MetamathQA}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
STEPS="${STEPS:-200}"
TURN="${TURN:-5}"
CONFIG="${CONFIG:-base}"
TRAIN_GROUPS="${TRAIN_GROUPS:-8}"
VAL_GROUPS="${VAL_GROUPS:-128}"
RESPONSE_LEN="${RESPONSE_LEN:-512}"  # Set larger for math-heavy envs (e.g., 1024 for MetaMathQA, 4096 for AIME24)

# Derive model name for checkpoint directory
if [[ "${MODEL_PATH}" == *"llama"* ]]; then
  MODEL_NAME="llama32_3b"
elif [[ "${MODEL_PATH}" == *"Qwen2.5-3B"* ]]; then
  MODEL_NAME="qwen25_3b"
elif [[ "${MODEL_PATH}" == *"Qwen2.5-7B"* ]]; then
  MODEL_NAME="qwen25_7b"
elif [[ "${MODEL_PATH}" == *"Qwen2.5-1.5B"* ]]; then
  MODEL_NAME="qwen25_1.5b"
else
  MODEL_NAME="model"
fi

# Lowercase env name for directory naming
ENV_LOWER=$(echo "${ENV}" | tr '[:upper:]' '[:lower:]')
EXPERIMENT="${MODEL_NAME}_${ENV_LOWER}_${TURN}turn_${STEPS}steps"
CKPT_DIR="/projects/bfea/lliu22/ragen_checkpoints/${EXPERIMENT}"
mkdir -p "${CKPT_DIR}"

echo "[INFO] ============================================"
echo "[INFO] Environment: ${ENV}"
echo "[INFO] Model:       ${MODEL_PATH}"
echo "[INFO] Steps:       ${STEPS}"
echo "[INFO] Max turn:    ${TURN}"
echo "[INFO] Config:      ${CONFIG}"
echo "[INFO] Experiment:  ${EXPERIMENT}"
echo "[INFO] Checkpoint:  ${CKPT_DIR}"
echo "[INFO] ============================================"

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  python train.py \
    --config-name="${CONFIG}" \
    system.CUDA_VISIBLE_DEVICES="'0,1'" \
    trainer.n_gpus_per_node=2 \
    trainer.total_training_steps="${STEPS}" \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.resume_mode=disable \
    trainer.project_name=ufb_train \
    trainer.experiment_name="${EXPERIMENT}" \
    trainer.default_local_dir="${CKPT_DIR}" \
    model_path="${MODEL_PATH}" \
    +trainer.max_actor_ckpt_to_keep=4 \
    +trainer.max_critic_ckpt_to_keep=0 \
    +actor_rollout_ref.actor.checkpoint.contents=[model,optimizer,extra,hf_config] \
    micro_batch_size_per_gpu=2 \
    ppo_mini_batch_size=16 \
    agent_proxy.max_turn="${TURN}" \
    val_agent_proxy.max_turn="${TURN}" \
    es_manager.train.env_groups="${TRAIN_GROUPS}" \
    es_manager.train.env_configs.tags="[${ENV}]" \
    es_manager.train.env_configs.n_groups="[${TRAIN_GROUPS}]" \
    es_manager.val.env_groups="${VAL_GROUPS}" \
    es_manager.val.env_configs.tags="[${ENV}]" \
    es_manager.val.env_configs.n_groups="[${VAL_GROUPS}]" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.response_length="${RESPONSE_LEN}"
