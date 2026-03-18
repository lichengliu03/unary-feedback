#!/usr/bin/env bash
#
# Evaluation script for independent pass@k (not sequential multi-turn).
#
# This evaluates the standard pass@k where k samples are generated independently.
#

set -x
set -euo pipefail

# ====== 基础配置 ======
DEVICES="${DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES="${DEVICES}"
HYDRA_VISIBLE_DEVICES="'${DEVICES}'"
GPUS_PER_NODE=$(echo "${DEVICES}" | tr ',' '\n' | wc -l)
TP_SIZE=$(python3 -c "print(min(4, '${DEVICES}'.count(',') + 1))")

PYTHON_BIN="${PYTHON_BIN:-python3}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

# 激活环境
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate ragen 2>/dev/null || true
fi

# ====== Eval 参数 ======
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
MODEL_NAME="${MODEL_NAME:-qwen25_3b_base}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-}"  # e.g., "global_step200"

# If checkpoint is specified, update model path
if [[ -n "${CHECKPOINT_STEP}" ]]; then
  BASE_CKPT_DIR="/projects/bfea/lliu22/ragen_checkpoints"
  CHECKPOINT_DIR="${BASE_CKPT_DIR}/${MODEL_NAME}/${CHECKPOINT_STEP}"

  if [[ -d "${CHECKPOINT_DIR}" ]]; then
    # Point to the HuggingFace format directory
    MODEL_PATH="${CHECKPOINT_DIR}/actor/huggingface"
    echo "[INFO] Using checkpoint: ${MODEL_PATH}"
  else
    echo "[ERROR] Checkpoint not found: ${CHECKPOINT_DIR}"
    exit 1
  fi
fi

WANDB_PROJECT="${WANDB_PROJECT:-independent_passk_eval}"
NUM_SAMPLES="${NUM_SAMPLES:-512}"  # Number of independent samples per problem (max k)
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.95}"
K_VALUES="${K_VALUES:-1,2,4,8,16,32,64,128,256,512}"  # K values to compute

EVAL_GPU_MEMORY_UTIL="${EVAL_GPU_MEMORY_UTIL:-0.4}"
EVAL_RESPONSE_LENGTH="${EVAL_RESPONSE_LENGTH:-4096}"  # Allow longer responses for complex AIME problems
EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-8192}"
ROLLOUT_LOAD_FORMAT="${ROLLOUT_LOAD_FORMAT:-safetensors}"

EVAL_QUESTIONS="${EVAL_QUESTIONS:-512}"  # Number of problems to evaluate
DATASET_TAG="${DATASET_TAG:-MetamathQA}"  # Dataset to evaluate on (MetamathQA or AIME24)

# ====== 输出目录 ======
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EVAL_RUN_NAME="${EVAL_RUN_NAME:-eval_independent_passk_${MODEL_NAME}_${TIMESTAMP}}"
OUT_BASE="${RESULT_BASE_DIR:-${REPO_DIR}/eval_results}/independent_passk"
mkdir -p "${OUT_BASE}"
RUN_OUT_DIR="${OUT_BASE}/${EVAL_RUN_NAME}"
mkdir -p "${RUN_OUT_DIR}"

echo "[INFO] Independent Pass@k Evaluation"
echo "[INFO] Model: ${MODEL_PATH}"
echo "[INFO] Model name: ${MODEL_NAME}"
echo "[INFO] Output dir: ${RUN_OUT_DIR}"
echo "[INFO] Eval questions: ${EVAL_QUESTIONS}"
echo "[INFO] Samples per problem (max k): ${NUM_SAMPLES}"
echo "[INFO] K values to compute: ${K_VALUES}"
echo "[INFO] Temperature: ${TEMPERATURE}, Top-p: ${TOP_P}"

# ====== 运行评估 ======
eval_key="${EVAL_RUN_NAME}"

EVAL_CMD_OVERRIDES=(
  --config-name base
  trainer.experiment_name="${eval_key}"
  trainer.project_name="${WANDB_PROJECT}"
  actor_rollout_ref.model.path="${MODEL_PATH}"
  system.CUDA_VISIBLE_DEVICES="${HYDRA_VISIBLE_DEVICES}"
  trainer.n_gpus_per_node=${GPUS_PER_NODE}
  actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE}

  # Evaluation mode
  trainer.total_training_steps=0
  trainer.save_freq=-1
  trainer.test_freq=1

  # Environment config
  es_manager.train.env_groups=1
  es_manager.train.group_size=1
  es_manager.train.env_configs.tags=[${DATASET_TAG}]
  es_manager.train.env_configs.n_groups=[1]
  es_manager.val.env_groups=${EVAL_QUESTIONS}
  es_manager.val.group_size=1
  es_manager.val.env_configs.tags=[${DATASET_TAG}]
  es_manager.val.env_configs.n_groups=[${EVAL_QUESTIONS}]

  # Agent config - SINGLE TURN ONLY
  agent_proxy.max_turn=1
  val_agent_proxy.max_turn=1

  # Rollout config - STOCHASTIC SAMPLING
  actor_rollout_ref.rollout.temperature=${TEMPERATURE}
  actor_rollout_ref.rollout.top_p=${TOP_P}
  actor_rollout_ref.rollout.val_kwargs.do_sample=true
  actor_rollout_ref.rollout.val_kwargs.temperature=${TEMPERATURE}
  actor_rollout_ref.rollout.val_kwargs.top_p=${TOP_P}
  actor_rollout_ref.rollout.gpu_memory_utilization=${EVAL_GPU_MEMORY_UTIL}
  actor_rollout_ref.rollout.response_length=${EVAL_RESPONSE_LENGTH}
  actor_rollout_ref.rollout.max_model_len=${EVAL_MAX_MODEL_LEN}
  actor_rollout_ref.rollout.load_format=${ROLLOUT_LOAD_FORMAT}

  # Stop tokens to prevent infinite generation
  '+actor_rollout_ref.rollout.stop=["</think>","</answer>","<|im_end|>"]'

  # Number of samples and k values
  +num_samples_per_problem=${NUM_SAMPLES}
  +k_values_to_compute=[${K_VALUES}]

  # Output config
  +output.dir="${RUN_OUT_DIR}"
  +output.filename="independent_passk.json"
  +output.append_timestamp=false

  trainer.logger=[console]
)

echo "[EVAL] Starting independent pass@k evaluation..."

WANDB_PROJECT="${WANDB_PROJECT}" \
WANDB_NAME="${eval_key}" \
WANDB_RUN_ID="${eval_key}" \
${PYTHON_BIN} -m ufb.eval_independent_passk "${EVAL_CMD_OVERRIDES[@]}"

echo "[DONE] Evaluation complete!"
echo ""
echo "Results saved to: ${RUN_OUT_DIR}"
echo "  independent_passk.json: ${RUN_OUT_DIR}/independent_passk_results.json"
