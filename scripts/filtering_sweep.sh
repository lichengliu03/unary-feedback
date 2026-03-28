#!/bin/bash
# Filtering Sweep: run the same training setup with multiple rollout-filter presets.
#
# Supported filter modes:
#   - nofilter
#   - top_k
#   - top_p
#
# Usage:
#   bash scripts/filtering_sweep.sh
#   FILTERS=nofilter bash scripts/filtering_sweep.sh
#   FILTERS=top_k,top_p ENV_TAG=SimpleSokoban bash scripts/filtering_sweep.sh
#
# By default this script runs all three filters in sequence:
#   nofilter -> top_k -> top_p
#
# Timing summary:
#   ESTIMATE_STEPS=200 bash scripts/filtering_sweep.sh
#   SUMMARY_LOG_FILE=outputs/logs/my_summary.log bash scripts/filtering_sweep.sh

set -euo pipefail
set +x

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}"

mkdir -p outputs/checkpoints
mkdir -p outputs/logs

export PYTHONPATH="${PROJECT_DIR}/verl:${PROJECT_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=ERROR
export VLLM_LOG_LEVEL=WARNING
export TRANSFORMERS_VERBOSITY=warning
export HF_HUB_VERBOSITY=error
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0

# ---- configurable ----
ENV_TAG="${ENV_TAG:-MetamathQA}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
STEPS="${STEPS:-200}"
NGPUS="${NGPUS:-1}"
SAVE_FREQ="${SAVE_FREQ:-50}"
FILTERS="${FILTERS:-all}"
ESTIMATE_STEPS="${ESTIMATE_STEPS:-200}"
TEST_FREQ="${TEST_FREQ:-10}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
TRAIN_ENV_GROUPS="${TRAIN_ENV_GROUPS:-8}"
TRAIN_GROUP_SIZE="${TRAIN_GROUP_SIZE:-16}"
VAL_ENV_GROUPS="${VAL_ENV_GROUPS:-512}"
VAL_GROUP_SIZE="${VAL_GROUP_SIZE:-1}"

# These defaults match docs/rollout_filtering.md.
TOP_K_VALUE="${TOP_K_VALUE:-0.25}"
TOP_P_VALUE="${TOP_P_VALUE:-0.9}"
TOP_P_SELECTION_EPS="${TOP_P_SELECTION_EPS:-0.01}"
TRANSFER_TO_HF="${TRANSFER_TO_HF:-1}"
DELETE_LOCAL_AFTER_TRANSFER="${DELETE_LOCAL_AFTER_TRANSFER:-1}"
HF_PRIVATE_REPO="${HF_PRIVATE_REPO:-1}"
HF_NAMESPACE="${HF_NAMESPACE:-}"
HF_REPO_PREFIX="${HF_REPO_PREFIX:-ufb}"
KEEP_LOCAL_TRANSFER_EXPORTS="${KEEP_LOCAL_TRANSFER_EXPORTS:-0}"

EXPERIMENT_BASE="${EXPERIMENT:-exp1_${ENV_TAG}}"
CKPT_ROOT="${CKPT_DIR:-${PROJECT_DIR}/outputs/checkpoints/${EXPERIMENT_BASE}}"
LOG_FILE_BASE="${LOG_FILE:-${PROJECT_DIR}/outputs/logs/${EXPERIMENT_BASE}.log}"
SUMMARY_LOG_FILE="${SUMMARY_LOG_FILE:-${PROJECT_DIR}/outputs/logs/${EXPERIMENT_BASE}_filtering_sweep_summary.log}"

ALL_FILTERS=("nofilter" "top_k" "top_p")
SELECTED_FILTERS=()
FILTER_ARGS=()
FILTER_SUMMARY=""

# Build CUDA device list: "0" for 1 GPU, "0,1" for 2 GPUs
CUDA_DEVICES=$(seq -s, 0 $((NGPUS - 1)))

# Set max_turn based on environment type:
#   Single-turn (MetamathQA, Countdown): max_turn = max retries (5)
#   Multi-turn (SimpleSokoban, FrozenLake): max_turn = turns_per_attempt * retries (15)
case "${ENV_TAG}" in
  SimpleSokoban|LargerSokoban|FrozenLake)
    MAX_TURN="${MAX_TURN:-15}"
    ;;
  *)
    MAX_TURN="${MAX_TURN:-5}"
    ;;
esac

format_seconds() {
  local total_seconds="$1"
  if [[ "${total_seconds}" == "NA" || -z "${total_seconds}" ]]; then
    echo "NA"
    return
  fi
  total_seconds="${total_seconds%.*}"
  local hours=$((total_seconds / 3600))
  local minutes=$(((total_seconds % 3600) / 60))
  local seconds=$((total_seconds % 60))
  printf "%02d:%02d:%02d" "${hours}" "${minutes}" "${seconds}"
}

append_summary() {
  printf "%s\n" "$*" >> "${SUMMARY_LOG_FILE}"
}

transfer_experiment_checkpoints() {
  local filter_name="$1"
  local run_experiment="$2"
  local run_ckpt_dir="$3"
  local transfer_export_root="${PROJECT_DIR}/outputs/hf_transfer/${run_experiment}"

  if [[ "${TRANSFER_TO_HF}" != "1" ]]; then
    return 0
  fi

  echo "[INFO] Uploading experiment checkpoints to Hugging Face..."
  EXPERIMENT_DIR="${run_ckpt_dir}" \
    MODEL_PATH="${MODEL_PATH}" \
    HF_PRIVATE_REPO="${HF_PRIVATE_REPO}" \
    HF_NAMESPACE="${HF_NAMESPACE}" \
    HF_REPO_PREFIX="${HF_REPO_PREFIX}" \
    DELETE_LOCAL_AFTER_UPLOAD="${DELETE_LOCAL_AFTER_TRANSFER}" \
    KEEP_LOCAL_EXPORTS="${KEEP_LOCAL_TRANSFER_EXPORTS}" \
    EXPORT_ROOT="${transfer_export_root}" \
    bash scripts/transfer_experiment_to_hf.sh
}

extract_log_metrics() {
  local log_file="$1"
  local target_steps="$2"

  python - "${log_file}" "${target_steps}" <<'PY'
import pathlib
import re
import sys

log_path = pathlib.Path(sys.argv[1])
target_steps = int(sys.argv[2])

if not log_path.exists():
    print("NA\t0\tNA\tNA\tNA")
    raise SystemExit(0)

text = log_path.read_text(errors="ignore")

configured_steps = "NA"
configured_matches = re.findall(r"Total steps:\s*(\d+)", text)
if configured_matches:
    configured_steps = configured_matches[-1]

validation_seconds = "NA"
validation_matches = re.findall(r"validation generation time:\s*([0-9.]+)\s*seconds", text)
if validation_matches:
    validation_seconds = validation_matches[-1]

rows = {}
for line in text.splitlines():
    step_match = re.search(r"step:(\d+)", line)
    total_match = re.search(r"timing_s/total:([0-9.]+)", line)
    if step_match and total_match:
        rows[int(step_match.group(1))] = float(total_match.group(1))

if not rows:
    print(f"{configured_steps}\t0\t{validation_seconds}\tNA\tNA")
    raise SystemExit(0)

items = sorted(rows.items())
deltas = [items[0][1]]
for i in range(1, len(items)):
    deltas.append(items[i][1] - items[i - 1][1])

avg_excl_first = sum(deltas[1:]) / len(deltas[1:]) if len(deltas) > 1 else deltas[0]
estimated_seconds = avg_excl_first * target_steps

print(
    f"{configured_steps}\t{items[-1][0]}\t{validation_seconds}\t"
    f"{avg_excl_first:.3f}\t{estimated_seconds:.1f}"
)
PY
}

resolve_filters() {
  local filters_arg="$1"
  local raw_filters=()

  if [[ -z "${filters_arg}" || "${filters_arg}" == "all" ]]; then
    SELECTED_FILTERS=("${ALL_FILTERS[@]}")
    return
  fi

  IFS=',' read -r -a raw_filters <<< "${filters_arg}"
  SELECTED_FILTERS=()

  for filter_name in "${raw_filters[@]}"; do
    filter_name="${filter_name//[[:space:]]/}"
    case "${filter_name}" in
      nofilter|top_k|top_p)
        SELECTED_FILTERS+=("${filter_name}")
        ;;
      all)
        SELECTED_FILTERS=("${ALL_FILTERS[@]}")
        return
        ;;
      *)
        echo "[ERROR] Unknown filter mode: ${filter_name}" >&2
        echo "[ERROR] Supported filter modes: nofilter, top_k, top_p, all" >&2
        exit 1
        ;;
    esac
  done

  if [[ ${#SELECTED_FILTERS[@]} -eq 0 ]]; then
    echo "[ERROR] No valid filter modes were selected." >&2
    exit 1
  fi
}

build_filter_args() {
  local filter_name="$1"
  FILTER_ARGS=()

  case "${filter_name}" in
    nofilter)
      FILTER_SUMMARY="strategy=top_k value=1.0 type=std include_zero=True"
      FILTER_ARGS=(
        actor_rollout_ref.rollout.rollout_filter_strategy=top_k
        actor_rollout_ref.rollout.rollout_filter_value=1.0
        actor_rollout_ref.rollout.rollout_filter_ratio=1.0
        actor_rollout_ref.rollout.rollout_filter_type=std
        actor_rollout_ref.rollout.rollout_filter_include_zero=True
      )
      ;;
    top_k)
      FILTER_SUMMARY="strategy=top_k value=${TOP_K_VALUE} type=std include_zero=True"
      FILTER_ARGS=(
        actor_rollout_ref.rollout.rollout_filter_strategy=top_k
        actor_rollout_ref.rollout.rollout_filter_value="${TOP_K_VALUE}"
        actor_rollout_ref.rollout.rollout_filter_ratio="${TOP_K_VALUE}"
        actor_rollout_ref.rollout.rollout_filter_type=std
        actor_rollout_ref.rollout.rollout_filter_include_zero=True
      )
      ;;
    top_p)
      FILTER_SUMMARY="strategy=top_p value=${TOP_P_VALUE} type=std prob_mode=linear include_zero=False eps=${TOP_P_SELECTION_EPS}"
      FILTER_ARGS=(
        actor_rollout_ref.rollout.rollout_filter_strategy=top_p
        actor_rollout_ref.rollout.rollout_filter_value="${TOP_P_VALUE}"
        actor_rollout_ref.rollout.rollout_filter_ratio="${TOP_P_VALUE}"
        actor_rollout_ref.rollout.rollout_filter_type=std
        actor_rollout_ref.rollout.rollout_filter_top_p_prob_mode=linear
        actor_rollout_ref.rollout.rollout_filter_include_zero=False
        actor_rollout_ref.rollout.rollout_filter_selection_eps="${TOP_P_SELECTION_EPS}"
      )
      ;;
    *)
      echo "[ERROR] Unsupported filter mode: ${filter_name}" >&2
      exit 1
      ;;
  esac
}

append_log_suffix() {
  local log_path="$1"
  local suffix="$2"

  if [[ "${log_path}" == *.log ]]; then
    printf '%s_%s.log' "${log_path%.log}" "${suffix}"
  else
    printf '%s_%s.log' "${log_path}" "${suffix}"
  fi
}

run_one_filter() {
  local filter_name="$1"
  local run_experiment="${EXPERIMENT_BASE}_${filter_name}"
  local run_ckpt_dir="${CKPT_ROOT}/${filter_name}"
  local run_log_file
  local status
  local run_status=0
  local start_seconds
  local elapsed_seconds
  local configured_steps
  local logged_steps
  local validation_seconds
  local avg_train_step_seconds
  local estimated_seconds_from_log
  local logged_summary
  local estimated_hms
  local validation_hms
  local transfer_status="SKIPPED"

  build_filter_args "${filter_name}"
  run_log_file="$(append_log_suffix "${LOG_FILE_BASE}" "${filter_name}")"

  mkdir -p "${run_ckpt_dir}"

  echo "[INFO] ============================================"
  echo "[INFO] Training"
  echo "[INFO] Env tag:     ${ENV_TAG}"
  echo "[INFO] Filter:      ${filter_name}"
  echo "[INFO] Filter cfg:  ${FILTER_SUMMARY}"
  echo "[INFO] GPUs:        ${NGPUS} (${CUDA_DEVICES})"
  echo "[INFO] Model:       ${MODEL_PATH}"
  echo "[INFO] Steps:       ${STEPS}"
  echo "[INFO] Test freq:   ${TEST_FREQ}"
  echo "[INFO] Val before:  ${VAL_BEFORE_TRAIN}"
  echo "[INFO] Max turn:    ${MAX_TURN}"
  echo "[INFO] Train envs:  ${TRAIN_ENV_GROUPS} x ${TRAIN_GROUP_SIZE}"
  echo "[INFO] Val envs:    ${VAL_ENV_GROUPS} x ${VAL_GROUP_SIZE}"
  echo "[INFO] Experiment:  ${run_experiment}"
  echo "[INFO] Checkpoint:  ${run_ckpt_dir}"
  echo "[INFO] Log file:    ${run_log_file}"
  echo "[INFO] ============================================"

  start_seconds="${SECONDS:-0}"

  if python train.py \
      --config-name=base \
      system.CUDA_VISIBLE_DEVICES="'${CUDA_DEVICES}'" \
      trainer.n_gpus_per_node="${NGPUS}" \
      trainer.total_training_steps="${STEPS}" \
      trainer.save_freq="${SAVE_FREQ}" \
      trainer.test_freq="${TEST_FREQ}" \
      trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
      trainer.resume_mode=disable \
      trainer.project_name=ufb_train \
      trainer.experiment_name="${run_experiment}" \
      trainer.default_local_dir="${run_ckpt_dir}" \
      model_path="${MODEL_PATH}" \
      ppo_micro_batch_size_per_gpu=4 \
      log_prob_micro_batch_size_per_gpu=16 \
      ppo_mini_batch_size=32 \
      actor_rollout_ref.actor.entropy_coeff=0.001 \
      agent_proxy.max_turn="${MAX_TURN}" \
      val_agent_proxy.max_turn="${MAX_TURN}" \
      actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
      actor_rollout_ref.rollout.max_model_len=8192 \
      actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
      es_manager.train.env_groups="${TRAIN_ENV_GROUPS}" \
      es_manager.train.group_size="${TRAIN_GROUP_SIZE}" \
      "es_manager.train.env_configs.tags=[${ENV_TAG}]" \
      "es_manager.train.env_configs.n_groups=[${TRAIN_ENV_GROUPS}]" \
      es_manager.val.env_groups="${VAL_ENV_GROUPS}" \
      es_manager.val.group_size="${VAL_GROUP_SIZE}" \
      "es_manager.val.env_configs.tags=[${ENV_TAG}]" \
      "es_manager.val.env_configs.n_groups=[${VAL_ENV_GROUPS}]" \
      +trainer.max_actor_ckpt_to_keep=4 \
      +trainer.max_critic_ckpt_to_keep=0 \
      "+actor_rollout_ref.actor.checkpoint.contents=[model,optimizer,extra,hf_config]" \
      "${FILTER_ARGS[@]}" \
      2>&1 | tee "${run_log_file}"; then
    status="OK"
  else
    status="FAILED"
    run_status=1
  fi

  elapsed_seconds=$((SECONDS - start_seconds))
  IFS=$'\t' read -r configured_steps logged_steps validation_seconds avg_train_step_seconds estimated_seconds_from_log < <(
    extract_log_metrics "${run_log_file}" "${ESTIMATE_STEPS}"
  )
  logged_summary="${logged_steps}/${configured_steps}"
  estimated_hms="$(format_seconds "${estimated_seconds_from_log}")"
  validation_hms="$(format_seconds "${validation_seconds}")"

  echo "[INFO] Result:     ${status}"
  echo "[INFO] Logged:     ${logged_summary} training steps"
  echo "[INFO] Wall time:   ${elapsed_seconds}s ($(format_seconds "${elapsed_seconds}"))"
  echo "[INFO] Train step:  ${avg_train_step_seconds}s avg"
  echo "[INFO] Estimate:    ${estimated_hms} (${estimated_seconds_from_log}s) for ${ESTIMATE_STEPS} steps"
  echo "[INFO] Validation:  ${validation_hms}"
  echo "[INFO] Log file:    ${run_log_file}"
  echo "[SUMMARY] ${filter_name} | steps ${logged_summary} | train_step ${avg_train_step_seconds}s | est_${ESTIMATE_STEPS} ${estimated_hms} | val ${validation_hms} | ${status}"

  append_summary "[${filter_name}]"
  append_summary "status: ${status}"
  append_summary "filter_config: ${FILTER_SUMMARY}"
  append_summary "steps_logged: ${logged_summary}"
  append_summary "wall_time: $(format_seconds "${elapsed_seconds}") (${elapsed_seconds}s)"
  append_summary "avg_train_step: ${avg_train_step_seconds}s"
  append_summary "est_${ESTIMATE_STEPS}_train: ${estimated_hms} (${estimated_seconds_from_log}s)"
  append_summary "validation: ${validation_hms}"
  append_summary "train_log: ${run_log_file}"
  append_summary "checkpoint_dir: ${run_ckpt_dir}"

  if [[ "${run_status}" -eq 0 ]]; then
    if transfer_experiment_checkpoints "${filter_name}" "${run_experiment}" "${run_ckpt_dir}"; then
      transfer_status="OK"
    else
      transfer_status="FAILED"
      run_status=1
    fi
  fi

  echo "[INFO] Transfer:    ${transfer_status}"
  append_summary "transfer_to_hf: ${transfer_status}"
  if [[ "${TRANSFER_TO_HF}" == "1" ]]; then
    append_summary "hf_repo_prefix: ${HF_REPO_PREFIX}"
    append_summary "hf_private_repo: ${HF_PRIVATE_REPO}"
  fi
  append_summary

  return "${run_status}"
}

resolve_filters "${FILTERS}"

echo "[INFO] Selected filters: ${SELECTED_FILTERS[*]}"
echo "[INFO] Checkpoint root:  ${CKPT_ROOT}"
echo "[INFO] Log file base:    ${LOG_FILE_BASE}"
echo "[INFO] Estimate to:     ${ESTIMATE_STEPS} steps"
echo "[INFO] Summary log:     ${SUMMARY_LOG_FILE}"
echo "[INFO] Transfer to HF:  ${TRANSFER_TO_HF}"
echo "[INFO] Delete local:    ${DELETE_LOCAL_AFTER_TRANSFER}"

: > "${SUMMARY_LOG_FILE}"
append_summary "Filtering Sweep summary"
append_summary "Env tag: ${ENV_TAG}"
append_summary "Selected filters: ${SELECTED_FILTERS[*]}"
append_summary "GPUs: ${NGPUS} (${CUDA_DEVICES})"
append_summary "Steps per filter: ${STEPS}"
append_summary "Save frequency: ${SAVE_FREQ}"
append_summary "Test frequency: ${TEST_FREQ}"
append_summary "Val before train: ${VAL_BEFORE_TRAIN}"
append_summary "Train env groups: ${TRAIN_ENV_GROUPS}"
append_summary "Train group size: ${TRAIN_GROUP_SIZE}"
append_summary "Val env groups: ${VAL_ENV_GROUPS}"
append_summary "Val group size: ${VAL_GROUP_SIZE}"
append_summary "Estimate target: ${ESTIMATE_STEPS}"
append_summary "Checkpoint root: ${CKPT_ROOT}"
append_summary "Transfer to HF: ${TRANSFER_TO_HF}"
append_summary "Delete local after transfer: ${DELETE_LOCAL_AFTER_TRANSFER}"
append_summary "Generated at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
append_summary

overall_status=0

for filter_name in "${SELECTED_FILTERS[@]}"; do
  if ! run_one_filter "${filter_name}"; then
    overall_status=1
  fi
done

exit "${overall_status}"
