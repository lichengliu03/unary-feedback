#!/bin/bash
#SBATCH -J no_feedback_train
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

echo "[INFO] Launching NO FEEDBACK training on ${SLURM_NODELIST}"

TRAIN_MAX_TURN="${TRAIN_MAX_TURN:-5}"
EVAL_MAX_TURN="${EVAL_MAX_TURN:-5}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"

# Checkpoint directory: model_turns_steps format
if [[ "${MODEL_PATH}" == *"llama"* ]]; then
  MODEL_NAME="llama32_3b"
elif [[ "${MODEL_PATH}" == *"Qwen"* ]] || [[ "${MODEL_PATH}" == *"qwen"* ]]; then
  MODEL_NAME="qwen25_3b"
else
  MODEL_NAME="unknown"
fi

CKPT_DIR="/projects/bfea/lliu22/ragen_checkpoints/${MODEL_NAME}_${TRAIN_MAX_TURN}turn_200steps_no_feedback"
mkdir -p "${CKPT_DIR}"

echo "[INFO] Model: ${MODEL_PATH}"
echo "[INFO] Max turn: ${TRAIN_MAX_TURN}"
echo "[INFO] Checkpoint dir: ${CKPT_DIR}"
echo "[INFO] Using NO FEEDBACK environment (empty observation on incorrect answers)"

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  python train.py \
    --config-name=train_no_feedback \
    system.CUDA_VISIBLE_DEVICES="'0,1'" \
    trainer.n_gpus_per_node=2 \
    trainer.total_training_steps=200 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.resume_mode=disable \
    trainer.project_name=ufb_train_no_feedback \
    trainer.experiment_name="${MODEL_NAME}_${TRAIN_MAX_TURN}turn_200steps_no_feedback" \
    trainer.default_local_dir="${CKPT_DIR}" \
    model_path="${MODEL_PATH}" \
    +trainer.max_actor_ckpt_to_keep=4 \
    +trainer.max_critic_ckpt_to_keep=0 \
    +actor_rollout_ref.actor.checkpoint.contents=[model,optimizer,extra,hf_config] \
    micro_batch_size_per_gpu=2 \
    ppo_mini_batch_size=16 \
    agent_proxy.max_turn="${TRAIN_MAX_TURN}" \
    val_agent_proxy.max_turn="${EVAL_MAX_TURN}" \
    es_manager.val.env_groups=256 \
    es_manager.val.env_configs.n_groups=[256] \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.response_length=512
