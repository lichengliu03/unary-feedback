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
#   ENV=metamathqa MODEL=Qwen/Qwen2.5-3B-Instruct sbatch scripts/eval.sh
#   ENV=sokoban MODEL=/path/to/checkpoint sbatch scripts/eval.sh
#
# Environment variables:
#   ENV           - Environment name (REQUIRED, matches configs/envs/<ENV>.yaml)
#   ENV_TAG       - Optional env tag override for es_manager.val.env_configs.tags
#   MODEL         - Model path or checkpoint path (REQUIRED)
#   VAL_GROUPS    - Number of validation groups (default: 256)
#   EVAL_TURN     - Max turns during evaluation (default: from env yaml)
#   CONTEXT_WINDOW_MODE - Prompt context mode: full | limited_multi_turn | single_turn
#   MAX_CONTEXT_WINDOW - Context window size used when CONTEXT_WINDOW_MODE=limited_multi_turn
#   LOG_DIR       - Directory to save raw eval logs (default: logs/eval)
#   OUTPUT_JSON   - Optional explicit detailed eval JSON output path
#   PARAMS_JSON   - Optional explicit eval-params JSON output path
#   SUMMARY_JSON  - Optional explicit summary JSON path converted from the log
#   CONDITIONAL_JSON - Optional explicit conditional-success JSON output path
#   CONVERT_JSON  - Whether to auto-run convert_out_to_json.py for SUMMARY_JSON (default: 1)
#   COMPUTE_CONDITIONAL_SUCCESS - Whether to auto-run compute_conditional_success.py (default: 1)
#   FEEDBACK_MODE - Optional feedback preset: one-bit | no-feedback | specific
#   FEEDBACK_RANDOMIZE - Optional manual override for envs that support env_config.randomize_feedback
#   FIXED_FEEDBACK - Optional manual override for envs that support env_config.fixed_feedback
#   FEEDBACK_TARGET - Optional override target: auto | env_config | retry
# ============================================================

ENV="${ENV:=hotpotqa}"
ENV_TAG="${ENV_TAG:=HotpotQA}"
MODEL="${MODEL:-/workspace/loaded_checkpoints/webshop_qwen25_3b_one_bit_2attempts_7turns/ufb_exp/webshop_qwen25_3b_one_bit_2attempts_7turns/global_step_200/actor/hf_merged}"
VAL_GROUPS="${VAL_GROUPS:-1024}"
EVAL_TURN="${EVAL_TURN:-}"  # empty = use env yaml default
CONTEXT_WINDOW_MODE="${CONTEXT_WINDOW_MODE:-full}"
MAX_CONTEXT_WINDOW="${MAX_CONTEXT_WINDOW:--1}"
CUDA_DEVICES="${CUDA_DEVICES:-1}"
NGPUS="${NGPUS:-1}"
TP_SIZE="${TP_SIZE:-1}"
MODEL_BASENAME="$(basename -- "${MODEL}")"
LOG_DIR="${LOG_DIR:-logs/eval}"
CONVERT_JSON="${CONVERT_JSON:-1}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/eval_${ENV_TAG}_${MODEL_BASENAME}_${TIMESTAMP}.log"
OUTPUT_JSON="${OUTPUT_JSON:-eval_results_2/eval_${ENV_TAG}_${MODEL_BASENAME}_${TIMESTAMP}.json}"
PARAMS_JSON="${PARAMS_JSON:-${OUTPUT_JSON%.json}.params.json}"
SUMMARY_JSON="${SUMMARY_JSON:-${OUTPUT_JSON%.json}.summary.json}"
CONDITIONAL_JSON="${CONDITIONAL_JSON:-${OUTPUT_JSON%.json}.conditional_success.json}"
COMPUTE_CONDITIONAL_SUCCESS="${COMPUTE_CONDITIONAL_SUCCESS:-1}"
FEEDBACK_MODE="${FEEDBACK_MODE:-one-bit}"
FEEDBACK_TARGET="${FEEDBACK_TARGET:-auto}"
FEEDBACK_RANDOMIZE_IS_SET="${FEEDBACK_RANDOMIZE+x}"
FEEDBACK_RANDOMIZE="${FEEDBACK_RANDOMIZE:-}"
FIXED_FEEDBACK_IS_SET="${FIXED_FEEDBACK+x}"
# FIXED_FEEDBACK_IS_SET="1"
FIXED_FEEDBACK="${FIXED_FEEDBACK:-}"

if [[ -n "${FEEDBACK_MODE}" ]]; then
  case "${FEEDBACK_MODE}" in
    one-bit)
      FEEDBACK_RANDOMIZE="true"
      FEEDBACK_RANDOMIZE_IS_SET="1"
      ;;
    no-feedback)
      FEEDBACK_RANDOMIZE="false"
      FEEDBACK_RANDOMIZE_IS_SET="1"
      FIXED_FEEDBACK=""
      FIXED_FEEDBACK_IS_SET="1"
      ;;
    specific)
      if [[ -z "${FIXED_FEEDBACK_IS_SET}" ]]; then
        echo "[ERROR] FEEDBACK_MODE=specific requires FIXED_FEEDBACK to be set." >&2
        exit 1
      fi
      FEEDBACK_RANDOMIZE="false"
      FEEDBACK_RANDOMIZE_IS_SET="1"
      ;;
    *)
      echo "[ERROR] Unsupported FEEDBACK_MODE: ${FEEDBACK_MODE}" >&2
      echo "[ERROR] Use one of: one-bit, no-feedback, specific" >&2
      exit 1
      ;;
  esac
fi

if [[ "${FEEDBACK_TARGET}" == "auto" ]]; then
  FEEDBACK_TARGET="$(python - "${ENV_TAG}" <<'PY'
import sys
import yaml

env_tag = sys.argv[1]
with open("configs/envs.yaml", "r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}
cfg = (data.get("custom_envs") or {}).get(env_tag) or {}
print("retry" if cfg.get("retry") is not None else "env_config")
PY
)"
fi

if [[ "${FEEDBACK_TARGET}" != "env_config" && "${FEEDBACK_TARGET}" != "retry" ]]; then
  echo "[ERROR] Unsupported FEEDBACK_TARGET: ${FEEDBACK_TARGET}" >&2
  echo "[ERROR] Use one of: auto, env_config, retry" >&2
  exit 1
fi

if (( TP_SIZE > NGPUS )); then
  echo "[ERROR] TP_SIZE (${TP_SIZE}) cannot be larger than visible GPU count (${NGPUS})." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname "${OUTPUT_JSON}")"
mkdir -p "$(dirname "${PARAMS_JSON}")"
mkdir -p "$(dirname "${SUMMARY_JSON}")"
mkdir -p "$(dirname "${CONDITIONAL_JSON}")"

echo "[INFO] ============================================"
echo "[INFO] Evaluation"
echo "[INFO] Environment: ${ENV}"
echo "[INFO] Env tag:     ${ENV_TAG:-<from env yaml>}"
echo "[INFO] Model:       ${MODEL}"
echo "[INFO] CUDA:        ${CUDA_DEVICES}"
echo "[INFO] GPUs:        ${NGPUS}"
echo "[INFO] TP size:     ${TP_SIZE}"
echo "[INFO] Val groups:  ${VAL_GROUPS}"
echo "[INFO] Eval turn:   ${EVAL_TURN:-<from env yaml>}"
echo "[INFO] Context:     ${CONTEXT_WINDOW_MODE}"
echo "[INFO] Context win: ${MAX_CONTEXT_WINDOW}"
echo "[INFO] Feedback:    ${FEEDBACK_MODE:-<env default/manual overrides>}"
echo "[INFO] Feedback target: ${FEEDBACK_TARGET}"
echo "[INFO] Log file:    ${LOG_FILE}"
echo "[INFO] Output JSON: ${OUTPUT_JSON}"
echo "[INFO] Params JSON: ${PARAMS_JSON}"
echo "[INFO] Summary JSON: ${SUMMARY_JSON}"
echo "[INFO] Conditional JSON: ${CONDITIONAL_JSON}"
echo "[INFO] ============================================"

# Build optional turn override
TURN_OVERRIDE=()
if [[ -n "${EVAL_TURN}" ]]; then
  TURN_OVERRIDE+=("val_agent_proxy.max_turn=${EVAL_TURN}")
fi

TAG_OVERRIDE=()
if [[ -n "${ENV_TAG}" ]]; then
  TAG_OVERRIDE+=("es_manager.val.env_configs.tags=[${ENV_TAG}]")
fi

FEEDBACK_OVERRIDE=()
case "${FEEDBACK_TARGET}" in
  env_config)
    if [[ -n "${FEEDBACK_RANDOMIZE_IS_SET}" ]]; then
      FEEDBACK_OVERRIDE+=("++custom_envs.${ENV_TAG}.env_config.randomize_feedback=${FEEDBACK_RANDOMIZE}")
    fi
    if [[ -n "${FIXED_FEEDBACK_IS_SET}" ]]; then
      FEEDBACK_OVERRIDE+=("++custom_envs.${ENV_TAG}.env_config.fixed_feedback=\"${FIXED_FEEDBACK}\"")
    fi
    ;;
  retry)
    if [[ -n "${FEEDBACK_RANDOMIZE_IS_SET}" ]]; then
      FEEDBACK_OVERRIDE+=("++custom_envs.${ENV_TAG}.retry.randomize_feedback=${FEEDBACK_RANDOMIZE}")
    fi
    if [[ -n "${FIXED_FEEDBACK_IS_SET}" ]]; then
      FEEDBACK_OVERRIDE+=("++custom_envs.${ENV_TAG}.retry.fixed_feedback=\"${FIXED_FEEDBACK}\"")
    fi
    ;;
esac

python train.py \
  --config-name="envs/${ENV}" \
  system.CUDA_VISIBLE_DEVICES="'${CUDA_DEVICES}'" \
  trainer.n_gpus_per_node="${NGPUS}" \
  trainer.total_training_steps=0 \
  trainer.save_freq=-1 \
  trainer.test_freq=1 \
  trainer.eval_output_json="${OUTPUT_JSON}" \
  trainer.eval_params_json="${PARAMS_JSON}" \
  trainer.eval_log_file="${LOG_FILE}" \
  trainer.eval_summary_json="${SUMMARY_JSON}" \
  trainer.eval_env_name="${ENV}" \
  trainer.eval_env_tag="${ENV_TAG}" \
  trainer.project_name=ufb_eval \
  trainer.experiment_name="eval_${ENV_TAG}_${MODEL_BASENAME}" \
  model_path="${MODEL}" \
  ++micro_batch_size_per_gpu=2 \
  ppo_mini_batch_size=16 \
  es_manager.val.env_groups="${VAL_GROUPS}" \
  es_manager.val.env_configs.n_groups="[${VAL_GROUPS}]" \
  agent_proxy.context_window_mode="${CONTEXT_WINDOW_MODE}" \
  agent_proxy.max_context_window="${MAX_CONTEXT_WINDOW}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
  "${TAG_OVERRIDE[@]}" \
  "${FEEDBACK_OVERRIDE[@]}" \
  "${TURN_OVERRIDE[@]}" \
  2>&1 | tee "${LOG_FILE}"

if [[ "${CONVERT_JSON}" == "1" ]]; then
  python scripts/utils/convert_out_to_json.py "${LOG_FILE}" "${SUMMARY_JSON}"
  echo "[INFO] Converted eval log to summary JSON: ${SUMMARY_JSON}"
fi

if [[ "${COMPUTE_CONDITIONAL_SUCCESS}" == "1" ]]; then
  python scripts/utils/compute_conditional_success.py "${SUMMARY_JSON}" "" "${CONDITIONAL_JSON}"
  echo "[INFO] Computed conditional success JSON: ${CONDITIONAL_JSON}"
fi
