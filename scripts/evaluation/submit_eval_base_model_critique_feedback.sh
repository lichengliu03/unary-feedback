#!/usr/bin/env bash

#SBATCH --job-name=eval_base_critique
#SBATCH --account=bfea-delta-gpu
#SBATCH --partition=gpuH200x8-interactive
#SBATCH --gres=gpu:h200:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH -o slurm-eval_base_critique-%j.out
#SBATCH -e slurm-eval_base_critique-%j.err

set -x
set -euo pipefail

# ====== 配置 ======
DEVICES="${DEVICES:-0,1}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
MODEL_NAME="${MODEL_NAME:-qwen25_3b_base_critique}"
EVAL_QUESTIONS="${EVAL_QUESTIONS:-1024}"
EVAL_MAX_TURN="${EVAL_MAX_TURN:-5}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"
EVAL_GPU_MEMORY_UTIL="${EVAL_GPU_MEMORY_UTIL:-0.5}"

echo "[INFO] ===== 评估基础模型 (Critique Environment) ====="
echo "[INFO] 模型路径: ${MODEL_PATH}"
echo "[INFO] 模型名称: ${MODEL_NAME}"
echo "[INFO] 评估题目数: ${EVAL_QUESTIONS}"
echo "[INFO] 最大轮数: ${EVAL_MAX_TURN}"
echo "[INFO] 批处理大小: ${EVAL_BATCH_SIZE}"
echo "[INFO] GPU数量: $(echo ${DEVICES} | tr ',' '\n' | wc -l)"
echo "[INFO] 使用 Critique 环境"

# 运行评估
DEVICES="${DEVICES}" \
MODEL_PATH="${MODEL_PATH}" \
MODEL_NAME="${MODEL_NAME}" \
EVAL_QUESTIONS="${EVAL_QUESTIONS}" \
EVAL_MAX_TURN="${EVAL_MAX_TURN}" \
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
EVAL_GPU_MEMORY_UTIL="${EVAL_GPU_MEMORY_UTIL}" \
bash scripts/evaluation/eval_base_model_critique_feedback.sh
