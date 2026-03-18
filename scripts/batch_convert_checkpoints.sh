#!/bin/bash
#SBATCH --job-name=convert_ckpts
#SBATCH --account=bfea-delta-gpu
#SBATCH --output=slurm-convert_ckpts-%j.out
#SBATCH --error=slurm-convert_ckpts-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --partition=gpuA40x4
#SBATCH --gpus-per-node=1

set -euo pipefail

# Activate conda environment
if [[ -f ~/miniconda3/etc/profile.d/conda.sh ]]; then
    source ~/miniconda3/etc/profile.d/conda.sh
fi
conda activate ragen

BASE_CKPT_DIR="/projects/bfea/lliu22/ragen_checkpoints"
BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
WORLD_SIZE=2

# List of checkpoints to convert
CKPTS=(
    "qwen25_3b_1turn_200steps/global_step_50"
    "qwen25_3b_1turn_200steps/global_step_100"
    "qwen25_3b_1turn_200steps/global_step_150"
    "qwen25_3b_1turn_200steps/global_step_200"
    "qwen25_3b_5turn_200steps/global_step_50"
    "qwen25_3b_5turn_200steps/global_step_100"
    "qwen25_3b_5turn_200steps/global_step_150"
    "qwen25_3b_5turn_200steps/global_step_200"
)

echo "=========================================="
echo "Starting batch checkpoint conversion"
echo "Base model: ${BASE_MODEL}"
echo "World size: ${WORLD_SIZE}"
echo "Total checkpoints: ${#CKPTS[@]}"
echo "=========================================="
echo ""

for ckpt_path in "${CKPTS[@]}"; do
    echo "----------------------------------------"
    echo "Converting: ${ckpt_path}"
    echo "----------------------------------------"

    CHECKPOINT_DIR="${BASE_CKPT_DIR}/${ckpt_path}/actor"
    OUTPUT_DIR="${BASE_CKPT_DIR}/${ckpt_path}/actor/huggingface"

    # Check if checkpoint exists
    if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
        echo "ERROR: Checkpoint directory not found: ${CHECKPOINT_DIR}"
        continue
    fi

    # Check if already converted
    if [[ -f "${OUTPUT_DIR}/model.safetensors" ]] || [[ -f "${OUTPUT_DIR}/model-00001-of-00002.safetensors" ]]; then
        echo "SKIP: Already converted (model weights exist)"
        continue
    fi

    echo "Input:  ${CHECKPOINT_DIR}"
    echo "Output: ${OUTPUT_DIR}"

    # Run conversion
    python3 ../tools/convert_fsdp_to_hf.py \
        "${CHECKPOINT_DIR}" \
        "${OUTPUT_DIR}" \
        "${BASE_MODEL}" \
        ${WORLD_SIZE}

    if [[ $? -eq 0 ]]; then
        echo "SUCCESS: Converted ${ckpt_path}"
    else
        echo "ERROR: Failed to convert ${ckpt_path}"
    fi
    echo ""
done

echo "=========================================="
echo "Batch conversion completed"
echo "=========================================="
