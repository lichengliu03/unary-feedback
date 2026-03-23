#!/bin/bash
# Quick smoke test wrapper around exp1_train.sh.

set -u -o pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}"

mkdir -p outputs/logs

STEPS="${STEPS:-10}"
ESTIMATE_STEPS="${ESTIMATE_STEPS:-200}"
NGPUS="${NGPUS:-1}"
# ENV_TAGS_STR="${ENV_TAGS_STR:-Countdown SimpleSokoban FrozenLake MetamathQA}"
ENV_TAGS_STR="${ENV_TAGS_STR:-MetamathQA}"
SUMMARY_LOG_FILE="${SUMMARY_LOG_FILE:-${PROJECT_DIR}/outputs/logs/quick_test_summary_${NGPUS}gpu.log}"

read -r -a ENV_TAGS <<< "${ENV_TAGS_STR}"

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

echo "[INFO] ============================================"
echo "[INFO] Quick test via scripts/exp1_train.sh"
echo "[INFO] Envs:        ${ENV_TAGS[*]}"
echo "[INFO] GPUs:        ${NGPUS}"
echo "[INFO] Steps/env:   ${STEPS}"
echo "[INFO] Estimate to: ${ESTIMATE_STEPS} steps"
echo "[INFO] Summary log: ${SUMMARY_LOG_FILE}"
echo "[INFO] ============================================"

: > "${SUMMARY_LOG_FILE}"
append_summary "Quick test summary"
append_summary "Envs: ${ENV_TAGS[*]}"
append_summary "GPUs: ${NGPUS}"
append_summary "Steps per env: ${STEPS}"
append_summary "Estimate target: ${ESTIMATE_STEPS}"
append_summary "Generated at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
append_summary

for ENV_TAG in "${ENV_TAGS[@]}"; do
  LOG_FILE="${PROJECT_DIR}/outputs/logs/exp1_${ENV_TAG}.log"

  echo
  echo "[INFO] --------------------------------------------"
  echo "[INFO] Env:        ${ENV_TAG}"
  echo "[INFO] Command:    ENV_TAG=${ENV_TAG} NGPUS=${NGPUS} STEPS=${STEPS} SAVE_FREQ=-1 bash scripts/exp1_train.sh"
  echo "[INFO] --------------------------------------------"

  start_seconds="${SECONDS:-0}"

  if ENV_TAG="${ENV_TAG}" NGPUS="${NGPUS}" STEPS="${STEPS}" SAVE_FREQ=-1 bash scripts/exp1_train.sh; then
    status="OK"
  else
    status="FAILED"
  fi

  elapsed_seconds=$((SECONDS - start_seconds))
  IFS=$'\t' read -r configured_steps logged_steps validation_seconds avg_train_step_seconds estimated_seconds_from_log < <(
    extract_log_metrics "${LOG_FILE}" "${ESTIMATE_STEPS}"
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
  echo "[INFO] Log file:    ${LOG_FILE}"
  echo "[SUMMARY] ${ENV_TAG} | steps ${logged_summary} | train_step ${avg_train_step_seconds}s | est_${ESTIMATE_STEPS} ${estimated_hms} | val ${validation_hms} | ${status}"

  append_summary "[${ENV_TAG}]"
  append_summary "status: ${status}"
  append_summary "steps_logged: ${logged_summary}"
  append_summary "avg_train_step: ${avg_train_step_seconds}s"
  append_summary "est_${ESTIMATE_STEPS}_train: ${estimated_hms} (${estimated_seconds_from_log}s)"
  append_summary "validation: ${validation_hms}"
  append_summary "train_log: ${LOG_FILE}"
  append_summary
done
