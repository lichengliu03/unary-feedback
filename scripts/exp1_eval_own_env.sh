#!/bin/bash
# Evaluate each step-checkpoint model on its own environment only.
# Covers: countdown, simplesokoban, frozenlake at steps 50 / 100 / 150.
# All other settings mirror exp1_eval.sh.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}/verl:${PROJECT_DIR}:${PYTHONPATH:-}"
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

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "[ERROR] Neither python nor python3 is available." >&2
  exit 1
fi

# ============================================================
# Exp1 own-env evaluation:
#   each model is evaluated only on its own training environment.
#
# Default models (countdown / simplesokoban / frozenlake, steps 50/100/150):
#   loaded_checkpoints/exp1/{countdown,simplesokoban,frozenlake}_{50,100,150}
#
# Override with:
#   MODELS_STR="loaded_checkpoints/exp1/countdown_50 loaded_checkpoints/exp1/countdown_100"
#   MODEL=loaded_checkpoints/exp1/countdown_50
#   MODELS_FILE=scripts/my_models.txt
#
# Common overrides:
#   VAL_GROUPS_PER_ENV=1024
#   MAX_ATTEMPTS=5
#   FEEDBACK_MODE=one-bit
#   CUDA_DEVICES=0
#   TP_SIZE=1
#   RUN_NAME=my_trial
#   OUTPUT_ROOT=eval_results/exp1_own_env
# ============================================================

VAL_GROUPS_PER_ENV="${VAL_GROUPS_PER_ENV:-1024}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-5}"
MAX_TURN_OVERRIDE="${MAX_TURN_OVERRIDE:-${EVAL_TURN:-}}"
FEEDBACK_MODE="${FEEDBACK_MODE:-one-bit}"
FEEDBACK_TARGET="${FEEDBACK_TARGET:-auto}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
SHOW_EVAL_PROGRESS="${SHOW_EVAL_PROGRESS:-1}"
EVAL_PROGRESS_INTERVAL="${EVAL_PROGRESS_INTERVAL:-1}"
NGPUS="$(echo "${CUDA_DEVICES}" | awk -F',' '{print NF}')"
TP_SIZE="${TP_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-14000}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-14000}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-400}"
OUTPUT_ROOT="${OUTPUT_ROOT:-eval_results/exp1_own_env}"
LOG_ROOT="${LOG_ROOT:-logs/exp1_own_env_eval}"
RUN_NAME="${RUN_NAME:-}"
SPLIT_PER_ENV="${SPLIT_PER_ENV:-1}"
COMPUTE_CONDITIONAL_SUCCESS="${COMPUTE_CONDITIONAL_SUCCESS:-1}"

if (( TP_SIZE > NGPUS )); then
  echo "[ERROR] TP_SIZE (${TP_SIZE}) cannot be larger than visible GPU count (${NGPUS})." >&2
  exit 1
fi

FEEDBACK_RANDOMIZE_IS_SET="${FEEDBACK_RANDOMIZE+x}"
FEEDBACK_RANDOMIZE="${FEEDBACK_RANDOMIZE:-}"
FIXED_FEEDBACK_IS_SET="${FIXED_FEEDBACK+x}"
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

# ── helpers ────────────────────────────────────────────────

join_by() {
  local delimiter="$1"; shift
  local first=1
  for item in "$@"; do
    if (( first )); then printf "%s" "${item}"; first=0
    else printf "%s%s" "${delimiter}" "${item}"; fi
  done
}

format_elapsed_hms() {
  local total_seconds="$1"
  printf "%02d:%02d:%02d" \
    "$(( total_seconds / 3600 ))" \
    "$(( (total_seconds % 3600) / 60 ))" \
    "$(( total_seconds % 60 ))"
}

build_hydra_string_list() {
  local items=("$@")
  local result="["; local first=1
  for item in "${items[@]}"; do
    if (( first )); then first=0; else result+=","; fi
    result+="'${item}'"
  done
  result+="]"
  printf "%s" "${result}"
}

build_repeated_int_list() {
  local count="$1"; local value="$2"
  local result="["
  for (( idx=0; idx<count; idx++ )); do
    if (( idx > 0 )); then result+=","; fi
    result+="${value}"
  done
  result+="]"
  printf "%s" "${result}"
}

slugify_path() {
  local base
  base="$(basename -- "${1%/}")"
  base="${base// /_}"; base="${base//\//_}"; base="${base//:/_}"
  base="$(printf "%s" "${base}" | tr -cd '[:alnum:]_.-')"
  [[ -z "${base}" ]] && base="model"
  printf "%s" "${base}"
}

# Map a model path to its own env tag.
# Matches the basename fragment before the step suffix.
env_tag_for_model() {
  local path="$1"
  local base
  base="$(basename -- "${path%/}")"
  case "${base}" in
    countdown*)    printf "Countdown"    ;;
    simplesokoban*) printf "SimpleSokoban" ;;
    frozenlake*)   printf "FrozenLake"  ;;
    metamathqa*)   printf "MetamathQA"  ;;
    hotpotqa*)     printf "HotpotQA"    ;;
    webshop*)      printf "WebShop"     ;;
    *)
      echo "[ERROR] Cannot infer env tag from model path: ${path}" >&2
      exit 1
      ;;
  esac
}

turns_per_attempt_for_env() {
  case "$1" in
    MetamathQA|Countdown) printf "1" ;;
    SimpleSokoban|FrozenLake) printf "5" ;;
    *) echo "[ERROR] Unsupported env tag: $1" >&2; exit 1 ;;
  esac
}

max_actions_per_attempt_for_env() {
  case "$1" in
    SimpleSokoban|FrozenLake) printf "10" ;;
    *) echo "[ERROR] Unsupported env tag for max-actions: $1" >&2; exit 1 ;;
  esac
}

uses_attempt_retry_wrapper_for_env() {
  case "$1" in
    SimpleSokoban|FrozenLake) return 0 ;;
    *) return 1 ;;
  esac
}

attempt_source_for_env() {
  case "$1" in
    MetamathQA|Countdown) printf "turn" ;;
    SimpleSokoban|FrozenLake) printf "attempt_num" ;;
    *) echo "[ERROR] Unsupported env tag: $1" >&2; exit 1 ;;
  esac
}

build_feedback_overrides() {
  local env_tag="$1"
  UFB_ENV_TAG="${env_tag}" \
  FEEDBACK_MODE="${FEEDBACK_MODE}" \
  FEEDBACK_TARGET="${FEEDBACK_TARGET}" \
  FEEDBACK_RANDOMIZE_IS_SET="${FEEDBACK_RANDOMIZE_IS_SET:-}" \
  FEEDBACK_RANDOMIZE="${FEEDBACK_RANDOMIZE}" \
  FIXED_FEEDBACK_IS_SET="${FIXED_FEEDBACK_IS_SET:-}" \
  FIXED_FEEDBACK="${FIXED_FEEDBACK}" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
import yaml

tag = os.environ["UFB_ENV_TAG"]
feedback_mode = os.environ.get("FEEDBACK_MODE", "")
feedback_target_override = os.environ.get("FEEDBACK_TARGET", "auto")
feedback_randomize_is_set = bool(os.environ.get("FEEDBACK_RANDOMIZE_IS_SET", ""))
feedback_randomize = os.environ.get("FEEDBACK_RANDOMIZE", "")
fixed_feedback_is_set = bool(os.environ.get("FIXED_FEEDBACK_IS_SET", ""))
fixed_feedback = os.environ.get("FIXED_FEEDBACK", "")

if not feedback_mode:
    raise SystemExit(0)

with open("configs/envs.yaml", "r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}
custom_envs = data.get("custom_envs") or {}

cfg = custom_envs.get(tag)
if cfg is None:
    raise SystemExit(f"[ERROR] Unknown env tag: {tag}")

target = feedback_target_override
if target == "auto":
    target = "retry" if cfg.get("retry") is not None else "env_config"
if target not in {"env_config", "retry"}:
    raise SystemExit(f"[ERROR] Unsupported FEEDBACK_TARGET: {target}")

prefix = f"++custom_envs.{tag}.{target}"
if feedback_randomize_is_set:
    print(f"{prefix}.randomize_feedback={feedback_randomize}")
if fixed_feedback_is_set:
    print(f"{prefix}.fixed_feedback={json.dumps(fixed_feedback)}")
PY
}

# ── default model list ──────────────────────────────────────

DEFAULT_MODELS=(
  "${PROJECT_DIR}/loaded_checkpoints/exp1/countdown_50"
  "${PROJECT_DIR}/loaded_checkpoints/exp1/countdown_100"
  "${PROJECT_DIR}/loaded_checkpoints/exp1/countdown_150"
  "${PROJECT_DIR}/loaded_checkpoints/exp1/simplesokoban_50"
  "${PROJECT_DIR}/loaded_checkpoints/exp1/simplesokoban_100"
  "${PROJECT_DIR}/loaded_checkpoints/exp1/simplesokoban_150"
  "${PROJECT_DIR}/loaded_checkpoints/exp1/frozenlake_50"
  "${PROJECT_DIR}/loaded_checkpoints/exp1/frozenlake_100"
  "${PROJECT_DIR}/loaded_checkpoints/exp1/frozenlake_150"
)

declare -a MODELS=()
if [[ -n "${MODEL:-}" ]]; then
  MODELS=("${MODEL}")
elif [[ -n "${MODELS_STR:-}" ]]; then
  read -r -a MODELS <<< "${MODELS_STR}"
elif [[ -n "${MODELS_FILE:-}" ]]; then
  [[ ! -f "${MODELS_FILE}" ]] && { echo "[ERROR] MODELS_FILE not found: ${MODELS_FILE}" >&2; exit 1; }
  mapfile -t MODELS < <(grep -v '^[[:space:]]*#' "${MODELS_FILE}" | sed '/^[[:space:]]*$/d')
else
  for p in "${DEFAULT_MODELS[@]}"; do
    if [[ -d "${p}" ]]; then
      MODELS+=("${p}")
    else
      echo "[WARN] Skipping missing model directory: ${p}" >&2
    fi
  done
fi

[[ "${#MODELS[@]}" -eq 0 ]] && { echo "[ERROR] No models resolved for evaluation." >&2; exit 1; }

if [[ -n "${RUN_NAME}" ]]; then
  RUN_ROOT="${OUTPUT_ROOT}/${RUN_NAME}"
  RUN_LOG_ROOT="${LOG_ROOT}/${RUN_NAME}"
  RUN_NAME_LABEL="${RUN_NAME}"
else
  RUN_ROOT="${OUTPUT_ROOT}"
  RUN_LOG_ROOT="${LOG_ROOT}"
  RUN_NAME_LABEL="(root)"
fi

mkdir -p "${RUN_ROOT}" "${RUN_LOG_ROOT}"

echo "[INFO] ============================================"
echo "[INFO] Exp1 Own-Env Evaluation"
echo "[INFO] Run name:        ${RUN_NAME_LABEL}"
echo "[INFO] Models:          ${#MODELS[@]}"
echo "[INFO] Samples/env:     ${VAL_GROUPS_PER_ENV}"
echo "[INFO] Max attempts:    ${MAX_ATTEMPTS}"
echo "[INFO] Feedback mode:   ${FEEDBACK_MODE}"
echo "[INFO] CUDA:            ${CUDA_DEVICES}"
echo "[INFO] GPUs:            ${NGPUS}"
echo "[INFO] TP size:         ${TP_SIZE}"
echo "[INFO] Output root:     ${RUN_ROOT}"
echo "[INFO] Python:          ${PYTHON_BIN}"
echo "[INFO] ============================================"

RESULTS_TSV="${RUN_ROOT}/results.tsv"

for MODEL_PATH in "${MODELS[@]}"; do
  MODEL_START_TS="$(date +%s)"
  MODEL_SLUG="$(slugify_path "${MODEL_PATH}")"

  # Resolve this model's own env.
  ENV_TAG="$(env_tag_for_model "${MODEL_PATH}")"
  TURNS_PER_ATTEMPT="$(turns_per_attempt_for_env "${ENV_TAG}")"
  ATTEMPT_SOURCE="$(attempt_source_for_env "${ENV_TAG}")"
  ENV_TOTAL_TURN="$(( TURNS_PER_ATTEMPT * MAX_ATTEMPTS ))"
  EVAL_MAX_TURN="${MAX_TURN_OVERRIDE:-${ENV_TOTAL_TURN}}"

  ENV_TAGS_HYDRA="$(build_hydra_string_list "${ENV_TAG}")"
  N_GROUPS_HYDRA="$(build_repeated_int_list 1 "${VAL_GROUPS_PER_ENV}")"

  declare -a RETRY_BUDGET_OVERRIDE=()
  declare -a INVALID_ATTEMPT_OVERRIDE=()
  ENV_MAX_ATTEMPT_ARGS=("--env-max-attempt" "${ENV_TAG}=${MAX_ATTEMPTS}")
  ENV_ATTEMPT_SOURCE_ARGS=("--env-attempt-source" "${ENV_TAG}=${ATTEMPT_SOURCE}")

  if [[ "${ATTEMPT_SOURCE}" == "turn" ]]; then
    INVALID_ATTEMPT_OVERRIDE=("++custom_envs.${ENV_TAG}.count_invalid_as_attempt=true")
  fi
  if uses_attempt_retry_wrapper_for_env "${ENV_TAG}"; then
    MAX_ACTIONS_PER_ATTEMPT="$(max_actions_per_attempt_for_env "${ENV_TAG}")"
    RETRY_BUDGET_OVERRIDE=(
      "++custom_envs.${ENV_TAG}.retry.max_turns_per_attempt=${TURNS_PER_ATTEMPT}"
      "++custom_envs.${ENV_TAG}.retry.max_actions_per_attempt=${MAX_ACTIONS_PER_ATTEMPT}"
      "++custom_envs.${ENV_TAG}.retry.max_retry_attempts=${MAX_ATTEMPTS}"
    )
  fi

  mapfile -t FEEDBACK_OVERRIDE < <(build_feedback_overrides "${ENV_TAG}")

  MODEL_RUN_DIR="${RUN_ROOT}/${MODEL_SLUG}"
  MODEL_LOG_FILE="${RUN_LOG_ROOT}/${MODEL_SLUG}.log"
  COMBINED_JSON="${MODEL_RUN_DIR}/combined.json"
  PARAMS_JSON="${MODEL_RUN_DIR}/combined.params.json"
  SUMMARY_JSON="${MODEL_RUN_DIR}/combined.summary.json"
  CONDITIONAL_JSON="${MODEL_RUN_DIR}/combined.conditional_success.json"
  PER_ENV_SUMMARY_DIR="${MODEL_RUN_DIR}/by_env"

  mkdir -p "${MODEL_RUN_DIR}"

  echo
  echo "[INFO] --------------------------------------------"
  echo "[INFO] Model:           ${MODEL_PATH}"
  echo "[INFO] Model label:     ${MODEL_SLUG}"
  echo "[INFO] Own env:         ${ENV_TAG}"
  echo "[INFO] Max turn:        ${EVAL_MAX_TURN}"
  echo "[INFO] Combined JSON:   ${COMBINED_JSON}"
  echo "[INFO] Summary JSON:    ${SUMMARY_JSON}"
  echo "[INFO] Log file:        ${MODEL_LOG_FILE}"
  echo "[INFO] --------------------------------------------"

  "${PYTHON_BIN}" train.py \
    --config-name=base \
    system.CUDA_VISIBLE_DEVICES="'${CUDA_DEVICES}'" \
    trainer.n_gpus_per_node="${NGPUS}" \
    trainer.total_training_steps=0 \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.eval_output_json="${COMBINED_JSON}" \
    trainer.eval_params_json="${PARAMS_JSON}" \
    trainer.eval_log_file="${MODEL_LOG_FILE}" \
    trainer.eval_summary_json="${SUMMARY_JSON}" \
    trainer.eval_env_name="exp1_own_env" \
    trainer.eval_env_tag="${ENV_TAG}" \
    trainer.eval_show_progress="${SHOW_EVAL_PROGRESS}" \
    trainer.eval_progress_interval="${EVAL_PROGRESS_INTERVAL}" \
    trainer.project_name=ufb_exp1_own_env_eval \
    trainer.experiment_name="exp1_own_env_${MODEL_SLUG}" \
    model_path="${MODEL_PATH}" \
    ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    agent_proxy.max_turn="${EVAL_MAX_TURN}" \
    val_agent_proxy.max_turn="${EVAL_MAX_TURN}" \
    actor_rollout_ref.rollout.response_length="${RESPONSE_LENGTH}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
    es_manager.val.env_groups="${VAL_GROUPS_PER_ENV}" \
    es_manager.val.group_size=1 \
    "es_manager.val.env_configs.tags=${ENV_TAGS_HYDRA}" \
    "es_manager.val.env_configs.n_groups=${N_GROUPS_HYDRA}" \
    "${RETRY_BUDGET_OVERRIDE[@]}" \
    "${INVALID_ATTEMPT_OVERRIDE[@]}" \
    "${FEEDBACK_OVERRIDE[@]}" \
    2>&1 | tee "${MODEL_LOG_FILE}"

  if [[ "${SPLIT_PER_ENV}" == "1" ]]; then
    "${PYTHON_BIN}" scripts/utils/build_eval_summary.py \
      "${COMBINED_JSON}" \
      "${SUMMARY_JSON}" \
      --default-max-attempt "${MAX_ATTEMPTS}" \
      "${ENV_MAX_ATTEMPT_ARGS[@]}" \
      "${ENV_ATTEMPT_SOURCE_ARGS[@]}" \
      --per-env-summary-dir "${PER_ENV_SUMMARY_DIR}"
  else
    "${PYTHON_BIN}" scripts/utils/build_eval_summary.py \
      "${COMBINED_JSON}" \
      "${SUMMARY_JSON}" \
      --default-max-attempt "${MAX_ATTEMPTS}" \
      "${ENV_MAX_ATTEMPT_ARGS[@]}" \
      "${ENV_ATTEMPT_SOURCE_ARGS[@]}"
  fi

  if [[ "${COMPUTE_CONDITIONAL_SUCCESS}" == "1" ]]; then
    "${PYTHON_BIN}" scripts/utils/compute_conditional_success.py \
      "${SUMMARY_JSON}" \
      "" \
      "${CONDITIONAL_JSON}"

    if [[ "${SPLIT_PER_ENV}" == "1" && -d "${PER_ENV_SUMMARY_DIR}" ]]; then
      shopt -s nullglob
      for PER_ENV_SUMMARY in "${PER_ENV_SUMMARY_DIR}"/*.summary.json; do
        PER_ENV_CONDITIONAL="${PER_ENV_SUMMARY%.summary.json}.conditional_success.json"
        "${PYTHON_BIN}" scripts/utils/compute_conditional_success.py \
          "${PER_ENV_SUMMARY}" \
          "" \
          "${PER_ENV_CONDITIONAL}"
      done
      shopt -u nullglob
    fi
  fi

  MODEL_END_TS="$(date +%s)"
  MODEL_ELAPSED_SEC="$(( MODEL_END_TS - MODEL_START_TS ))"
  echo "[INFO] Model completed: ${MODEL_SLUG} (env=${ENV_TAG}) | elapsed=$(format_elapsed_hms "${MODEL_ELAPSED_SEC}")"
done

# Aggregate TSV (same format as exp1_eval.sh)
if [[ "${SPLIT_PER_ENV}" == "1" ]]; then
  "${PYTHON_BIN}" - "${RUN_ROOT}" "${RESULTS_TSV}" <<'PY'
import json, sys
from pathlib import Path

run_root = Path(sys.argv[1])
output_path = Path(sys.argv[2])

rows = []
for summary_path in sorted(run_root.glob("*/by_env/*.summary.json")):
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    entry = payload.get("base_model") or {}
    metadata = payload.get("metadata") or {}
    pass_at_k = entry.get("pass_at_k") or {}
    rows.append({
        "model_label": summary_path.parents[1].name,
        "env_tag": metadata.get("env_tag", summary_path.stem.replace(".summary", "")),
        "avg_success": entry.get("avg_success"),
        "pass@1": pass_at_k.get("pass@1"),
        "pass@2": pass_at_k.get("pass@2"),
        "pass@3": pass_at_k.get("pass@3"),
        "pass@4": pass_at_k.get("pass@4"),
        "pass@5": pass_at_k.get("pass@5"),
        "avg_reward": entry.get("avg_reward"),
        "avg_num_actions": entry.get("avg_num_actions"),
        "num_episodes": entry.get("num_episodes"),
        "model_path": metadata.get("model_path"),
        "summary_json": str(summary_path),
    })

fields = [
    "model_label", "env_tag", "avg_success",
    "pass@1", "pass@2", "pass@3", "pass@4", "pass@5",
    "avg_reward", "avg_num_actions", "num_episodes",
    "model_path", "summary_json",
]

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as f:
    f.write("\t".join(fields) + "\n")
    for row in rows:
        f.write("\t".join(str(row.get(field) or "") for field in fields) + "\n")

print(f"[INFO] Wrote result matrix TSV: {output_path}")
PY
fi

echo "[INFO] Finished Exp1 own-env evaluation."
echo "[INFO] Run directory: ${RUN_ROOT}"
[[ -f "${RESULTS_TSV}" ]] && echo "[INFO] Result matrix: ${RESULTS_TSV}"
