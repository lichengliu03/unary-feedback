#!/bin/bash
#SBATCH --partition=gpuH200x8
#SBATCH --account=bflz-delta-gpu
#SBATCH -J metamathqa_uf_rl
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH -t 24:00:00
#SBATCH -o /u/ylin30/unary-feedback/outputs/slurm/slurm-%x-%j.out
#SBATCH -e /u/ylin30/unary-feedback/outputs/slurm/slurm-%x-%j.err

set -euo pipefail

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load cuda/12.8 || true
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate ragen
fi

cd /u/ylin30/unary-feedback

mkdir -p outputs/slurm
mkdir -p outputs/checkpoints

export PYTHONPATH="/u/ylin30/unary-feedback/verl:/u/ylin30/unary-feedback:${PYTHONPATH:-}"
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

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
STEPS="${STEPS:-200}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-10}"
EXPERIMENT="${EXPERIMENT:-qwen25_3b_metamathqa_unary_feedback_pool_${STEPS}steps_5turns}"
CKPT_DIR="${CKPT_DIR:-/projects/bflz/${EXPERIMENT}}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"

mkdir -p "${CKPT_DIR}"

echo "[INFO] ============================================"
echo "[INFO] Environment: MetamathQA"
echo "[INFO] Model:       ${MODEL_PATH}"
echo "[INFO] Steps:       ${STEPS}"
echo "[INFO] Save freq:   ${SAVE_FREQ}"
echo "[INFO] Checkpoint:  ${CKPT_DIR}"
echo "[INFO] ============================================"

python train.py \
  --config-name="envs/metamathqa" \
  system.CUDA_VISIBLE_DEVICES="'${CUDA_DEVICES}'" \
  trainer.n_gpus_per_node=2 \
  trainer.total_training_steps="${STEPS}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.resume_mode=disable \
  trainer.project_name=ufb_train \
  trainer.experiment_name="${EXPERIMENT}" \
  trainer.default_local_dir="${CKPT_DIR}" \
  model_path="${MODEL_PATH}" \
  custom_envs.MetamathQA.env_config.randomize_feedback=true \
  +trainer.max_actor_ckpt_to_keep=4 \
  +trainer.max_critic_ckpt_to_keep=0 \
  +actor_rollout_ref.actor.checkpoint.contents=[model,optimizer,extra,hf_config] \
  micro_batch_size_per_gpu=2 \
  ppo_mini_batch_size=16 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
  actor_rollout_ref.rollout.max_model_len=8192 \
  actor_rollout_ref.rollout.max_num_batched_tokens=16384
