#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DRY_RUN="${DRY_RUN:-0}"

mapfile -t EXPERIMENT_SCRIPTS < <(
  find "${SCRIPT_DIR}" -maxdepth 1 -type f -name '*.sh' \
    ! -name '_*' \
    ! -name 'submit_all.sh' \
    | sort
)

if [ "${#EXPERIMENT_SCRIPTS[@]}" -eq 0 ]; then
  echo "[ERROR] No experiment scripts found in ${SCRIPT_DIR}"
  exit 1
fi

echo "[INFO] Found ${#EXPERIMENT_SCRIPTS[@]} experiment scripts."

for script in "${EXPERIMENT_SCRIPTS[@]}"; do
  base_name="$(basename "${script}")"
  if [ "${DRY_RUN}" = "1" ]; then
    echo "[DRY RUN] sbatch ${script}"
    continue
  fi

  echo "[INFO] Submitting ${base_name}"
  sbatch_output="$(sbatch "${script}")"
  echo "[INFO] ${sbatch_output}"
done
