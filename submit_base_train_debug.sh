#!/bin/bash
#SBATCH -J base_debug
#SBATCH -N 1
#SBATCH --partition=gpuH200x8-interactive
#SBATCH --account=bfea-delta-gpu
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH -t 00:30:00
#SBATCH -o /u/lliu22/unary-feedback/slurm-%x-%j.out
#SBATCH -e /u/lliu22/unary-feedback/slurm-%x-%j.err

set -euxo pipefail

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

echo "[INFO] Launching training on ${SLURM_NODELIST}"
echo "[DEBUG] PYTHONPATH=${PYTHONPATH}"
echo "[DEBUG] Current directory: $(pwd)"
echo "[DEBUG] Python version: $(python --version)"
echo "[DEBUG] Testing import..."
python -c "from ufb.trainer.agent_trainer import RayAgentTrainer; print('Import successful')"

TRAIN_MAX_TURN="${TRAIN_MAX_TURN:-5}"
EVAL_MAX_TURN="${EVAL_MAX_TURN:-5}"

echo "[DEBUG] Starting training with 2 steps for testing..."

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  python -c "import sys; sys.path.insert(0, '/u/lliu22/unary-feedback'); import train" \
    --config-name=base \
    system.CUDA_VISIBLE_DEVICES="0,1" \
    trainer.n_gpus_per_node=2 \
    trainer.total_training_steps=2 \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.project_name=base_debug \
    trainer.experiment_name=debug_test \
    micro_batch_size_per_gpu=2 \
    ppo_mini_batch_size=16 \
    agent_proxy.max_turn="${TRAIN_MAX_TURN}" \
    val_agent_proxy.max_turn="${EVAL_MAX_TURN}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.response_length=512
