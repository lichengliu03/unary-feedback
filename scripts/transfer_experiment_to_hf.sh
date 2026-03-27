#!/bin/bash
# Export every checkpoint in an experiment directory to Hugging Face, upload them,
# and optionally delete the local checkpoints afterwards.

set -euo pipefail
set +x

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}"

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN:-python}"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN:-python3}"
else
  echo "[ERROR] Neither python nor python3 was found in PATH."
  exit 1
fi

EXPERIMENT_DIR="${EXPERIMENT_DIR:-${CKPT_DIR:?Please set EXPERIMENT_DIR or CKPT_DIR to an experiment directory}}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
HF_PRIVATE_REPO="${HF_PRIVATE_REPO:-1}"
HF_NAMESPACE="${HF_NAMESPACE:-}"
HF_REPO_PREFIX="${HF_REPO_PREFIX:-ufb}"
HF_REPO_ID="${HF_REPO_ID:-}"
DELETE_LOCAL_AFTER_UPLOAD="${DELETE_LOCAL_AFTER_UPLOAD:-1}"
KEEP_LOCAL_EXPORTS="${KEEP_LOCAL_EXPORTS:-0}"
DRY_RUN="${DRY_RUN:-0}"

EXPERIMENT_DIR="$(cd "${EXPERIMENT_DIR}" && pwd)"
EXPERIMENT_PARENT="$(basename "$(dirname "${EXPERIMENT_DIR}")")"
EXPERIMENT_NAME="$(basename "${EXPERIMENT_DIR}")"
EXPORT_ROOT="${EXPORT_ROOT:-${PROJECT_DIR}/outputs/hf_transfer/${EXPERIMENT_PARENT}_${EXPERIMENT_NAME}}"

mapfile -t STEP_DIRS < <(find "${EXPERIMENT_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' | sort -V)
if [[ ${#STEP_DIRS[@]} -eq 0 ]]; then
  echo "[ERROR] No global_step_* directories found under ${EXPERIMENT_DIR}" >&2
  exit 1
fi

LATEST_STEP_DIR="${STEP_DIRS[$((${#STEP_DIRS[@]} - 1))]}"
LATEST_STEP_NAME="$(basename "${LATEST_STEP_DIR}")"

if [[ -z "${HF_NAMESPACE}" ]]; then
  HF_NAMESPACE="$("${PYTHON_BIN}" - <<'PY'
import sys
from huggingface_hub import HfApi

try:
    print(HfApi().whoami()["name"])
except Exception:
    print("[ERROR] Hugging Face auth not found. Please run `hf auth login` before uploading.", file=sys.stderr)
    raise
PY
)"
fi

if [[ -z "${HF_REPO_ID}" ]]; then
  REPO_SLUG="$(
    EXPERIMENT_PARENT="${EXPERIMENT_PARENT}" EXPERIMENT_NAME="${EXPERIMENT_NAME}" HF_REPO_PREFIX="${HF_REPO_PREFIX}" "${PYTHON_BIN}" - <<'PY'
import os
import re

parts = [os.environ["EXPERIMENT_PARENT"], os.environ["EXPERIMENT_NAME"]]
slug = "-".join(part for part in parts if part)
slug = slug.lower().replace("_", "-")
slug = re.sub(r"[^a-z0-9.-]+", "-", slug)
slug = re.sub(r"-+", "-", slug).strip("-")
print(f"{os.environ['HF_REPO_PREFIX']}-{slug}")
PY
  )"
  HF_REPO_ID="${HF_NAMESPACE}/${REPO_SLUG}"
fi

mkdir -p "${EXPORT_ROOT}"

echo "[INFO] ============================================"
echo "[INFO] Transfer experiment to Hugging Face"
echo "[INFO] Python:       ${PYTHON_BIN}"
echo "[INFO] Experiment:   ${EXPERIMENT_DIR}"
echo "[INFO] Base model:   ${MODEL_PATH}"
echo "[INFO] Repo:         ${HF_REPO_ID}"
echo "[INFO] Private:      ${HF_PRIVATE_REPO}"
echo "[INFO] Export root:  ${EXPORT_ROOT}"
echo "[INFO] Latest step:  ${LATEST_STEP_NAME}"
echo "[INFO] Dry run:      ${DRY_RUN}"
echo "[INFO] ============================================"

if [[ "${DRY_RUN}" == "1" ]]; then
  for step_dir in "${STEP_DIRS[@]}"; do
    step_name="$(basename "${step_dir}")"
    echo "[DRY RUN] export ${step_dir} -> ${EXPORT_ROOT}/${step_name}"
    echo "[DRY RUN] upload ${EXPORT_ROOT}/${step_name} -> ${HF_REPO_ID}/checkpoints/${step_name}"
    if [[ "${step_name}" == "${LATEST_STEP_NAME}" ]]; then
      echo "[DRY RUN] upload ${EXPORT_ROOT}/${step_name} -> ${HF_REPO_ID}/"
    fi
  done
  if [[ "${DELETE_LOCAL_AFTER_UPLOAD}" == "1" ]]; then
    echo "[DRY RUN] delete local experiment dir ${EXPERIMENT_DIR} after successful upload"
  fi
  exit 0
fi

for step_dir in "${STEP_DIRS[@]}"; do
  step_name="$(basename "${step_dir}")"
  export_dir="${EXPORT_ROOT}/${step_name}"

  echo "[INFO] Exporting ${step_name}..."
  PYTHON_BIN="${PYTHON_BIN}" \
    CKPT_DIR="${step_dir}" \
    MODEL_PATH="${MODEL_PATH}" \
    OUTPUT_DIR="${export_dir}" \
    bash scripts/transfer_to_hf.sh

  echo "[INFO] Uploading ${step_name} -> checkpoints/${step_name}"
  HF_REPO_ID="${HF_REPO_ID}" \
  HF_PRIVATE_REPO="${HF_PRIVATE_REPO}" \
  EXPORT_DIR="${export_dir}" \
  STEP_NAME="${step_name}" \
  PATH_IN_REPO="checkpoints/${step_name}" \
  "${PYTHON_BIN}" - <<'PY'
import os
from huggingface_hub import HfApi

api = HfApi()
api.create_repo(
    repo_id=os.environ["HF_REPO_ID"],
    repo_type="model",
    private=os.environ["HF_PRIVATE_REPO"] == "1",
    exist_ok=True,
)
api.upload_folder(
    folder_path=os.environ["EXPORT_DIR"],
    repo_id=os.environ["HF_REPO_ID"],
    repo_type="model",
    path_in_repo=os.environ["PATH_IN_REPO"],
    commit_message=f"Upload {os.environ['STEP_NAME']}",
)
PY

  if [[ "${step_name}" == "${LATEST_STEP_NAME}" ]]; then
    echo "[INFO] Uploading ${step_name} as repo root (latest checkpoint)"
    HF_REPO_ID="${HF_REPO_ID}" \
    HF_PRIVATE_REPO="${HF_PRIVATE_REPO}" \
    EXPORT_DIR="${export_dir}" \
    STEP_NAME="${step_name}" \
    "${PYTHON_BIN}" - <<'PY'
import os
from huggingface_hub import HfApi

api = HfApi()
api.create_repo(
    repo_id=os.environ["HF_REPO_ID"],
    repo_type="model",
    private=os.environ["HF_PRIVATE_REPO"] == "1",
    exist_ok=True,
)
api.upload_folder(
    folder_path=os.environ["EXPORT_DIR"],
    repo_id=os.environ["HF_REPO_ID"],
    repo_type="model",
    path_in_repo="",
    commit_message=f"Upload latest checkpoint {os.environ['STEP_NAME']} to repo root",
)
PY
  fi
done

if [[ -f "${EXPERIMENT_DIR}/latest_checkpointed_iteration.txt" ]]; then
  echo "[INFO] Uploading latest_checkpointed_iteration.txt"
  HF_REPO_ID="${HF_REPO_ID}" \
  LATEST_FILE="${EXPERIMENT_DIR}/latest_checkpointed_iteration.txt" \
  "${PYTHON_BIN}" - <<'PY'
import os
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj=os.environ["LATEST_FILE"],
    path_in_repo="latest_checkpointed_iteration.txt",
    repo_id=os.environ["HF_REPO_ID"],
    repo_type="model",
    commit_message="Upload latest checkpoint pointer",
)
PY
fi

if [[ "${DELETE_LOCAL_AFTER_UPLOAD}" == "1" ]]; then
  echo "[INFO] Deleting local experiment dir: ${EXPERIMENT_DIR}"
  rm -rf "${EXPERIMENT_DIR}"
fi

if [[ "${KEEP_LOCAL_EXPORTS}" != "1" ]]; then
  echo "[INFO] Deleting local export dir: ${EXPORT_ROOT}"
  rm -rf "${EXPORT_ROOT}"
fi

echo "[INFO] Uploaded repo: https://huggingface.co/${HF_REPO_ID}"
