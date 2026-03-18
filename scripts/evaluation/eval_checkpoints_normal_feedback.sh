#!/usr/bin/env bash

set -euo pipefail

# ====== 基础配置 ======
DEVICES="${DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES="${DEVICES}"
HYDRA_VISIBLE_DEVICES="'${DEVICES}'"
GPUS_PER_NODE=$(echo "${DEVICES}" | tr ',' '\n' | wc -l)
TP_SIZE=$(python3 -c "print(min(4, '${DEVICES}'.count(',') + 1))")

PYTHON_BIN="${PYTHON_BIN:-python3}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

# 激活环境
if [[ "${SKIP_BASHRC:-0}" != "1" ]]; then
  { source ~/.bashrc >/dev/null 2>&1 || true; } || true
fi
[[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]] && source "$HOME/miniconda3/etc/profile.d/conda.sh"

set +u
if command -v conda >/dev/null 2>&1; then
  if conda info --envs | grep -q 'ragen'; then
    if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "ragen" ]]; then
      eval "$(conda shell.bash hook)"
      conda activate ragen || true
    fi
  fi
fi
set -u

# ====== Eval 参数 ======
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
WANDB_PROJECT="${WANDB_PROJECT:-ragen_metamathqa}"
EVAL_MAX_TURN="${EVAL_MAX_TURN:-10}"
EVAL_GPU_MEMORY_UTIL="${EVAL_GPU_MEMORY_UTIL:-0.4}"
EVAL_RESPONSE_LENGTH="${EVAL_RESPONSE_LENGTH:-256}"
EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-2048}"
ROLLOUT_LOAD_FORMAT="${ROLLOUT_LOAD_FORMAT:-safetensors}"
ROLLOUT_LAYERED_SUMMON="${ROLLOUT_LAYERED_SUMMON:-1}"
DATASET_TAG="MetamathQA"
MAX_ACTIONS_PER_TRAJ="${MAX_ACTIONS_PER_TRAJ:-5}"

EVAL_QUESTIONS="${EVAL_QUESTIONS:-1024}"
EVAL_GROUP_SIZE="${EVAL_GROUP_SIZE:-1}"
EVAL_K_VALUES_ENV="${EVAL_K_VALUES:-1 2 4 6 8 10}"
read -r -a EVAL_K_VALUES <<< "${EVAL_K_VALUES_ENV}"

# ====== 批处理配置 ======
# 不要一次性创建所有环境，而是分批处理
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"  # 每批处理的环境数
if (( EVAL_QUESTIONS % EVAL_BATCH_SIZE != 0 )); then
  echo "[WARN] EVAL_QUESTIONS (${EVAL_QUESTIONS}) 不能被 EVAL_BATCH_SIZE (${EVAL_BATCH_SIZE}) 整除，将向上取整"
fi
EVAL_NUM_BATCHES=$(( (EVAL_QUESTIONS + EVAL_BATCH_SIZE - 1) / EVAL_BATCH_SIZE ))

echo "[INFO] 将 ${EVAL_QUESTIONS} 道题分成 ${EVAL_NUM_BATCHES} 批，每批 ${EVAL_BATCH_SIZE} 道题"

# ====== 指定已有的 checkpoint 目录 ======
CKPT_DIR="${CKPT_DIR:-/projects/bfea/lliu22/ragen_checkpoints/metamathqa_qwen25_3b_single_rl_slurm_20260105_095202}"
CKPT_STEPS_ENV="${CKPT_STEPS:-50 100 200}"
read -r -a CKPT_STEPS <<< "${CKPT_STEPS_ENV}"

# ====== 输出目录 ======
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EVAL_RUN_NAME="${EVAL_RUN_NAME:-eval_only_${TIMESTAMP}}"
OUT_BASE="${RESULT_BASE_DIR:-${REPO_DIR}/result}/new_experiments/metamathqa_single_round"
mkdir -p "${OUT_BASE}"
RUN_OUT_DIR="${OUT_BASE}/${EVAL_RUN_NAME}"
mkdir -p "${RUN_OUT_DIR}"

echo "[INFO] Checkpoint dir: ${CKPT_DIR}"
echo "[INFO] Output dir: ${RUN_OUT_DIR}"
echo "[INFO] Eval steps: ${CKPT_STEPS[*]}"
echo "[INFO] Eval questions: ${EVAL_QUESTIONS}, max_turn: ${EVAL_MAX_TURN}"

ROLLOUT_LAYERED_SUMMON_BOOL="False"
if [[ "${ROLLOUT_LAYERED_SUMMON}" == "1" || "${ROLLOUT_LAYERED_SUMMON}" == "true" ]]; then
  ROLLOUT_LAYERED_SUMMON_BOOL="True"
fi

run_eval() {
  local model_path="$1"
  local eval_key="$2"
  local out_dir="$3"
  mkdir -p "${out_dir}"
  echo "[EVAL] ${eval_key} -> ${model_path}"

  # 分批运行评估
  for (( batch=0; batch<EVAL_NUM_BATCHES; batch++ )); do
    local batch_start=$(( batch * EVAL_BATCH_SIZE ))
    local batch_end=$(( (batch + 1) * EVAL_BATCH_SIZE ))
    if (( batch_end > EVAL_QUESTIONS )); then
      batch_end=${EVAL_QUESTIONS}
    fi
    local batch_size=$(( batch_end - batch_start ))

    echo "[EVAL] 批次 $((batch+1))/${EVAL_NUM_BATCHES}: 题目 ${batch_start}-${batch_end} (共 ${batch_size} 题)"

    local batch_out_dir="${out_dir}/batch_${batch}"
    mkdir -p "${batch_out_dir}"

    EVAL_CMD_OVERRIDES=(
      --config-name eval
      trainer.experiment_name="${eval_key}_batch${batch}"
      trainer.project_name="${WANDB_PROJECT}"
      actor_rollout_ref.model.path="${model_path}"
      system.CUDA_VISIBLE_DEVICES="${HYDRA_VISIBLE_DEVICES}"
      trainer.n_gpus_per_node=${GPUS_PER_NODE}
      actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE}
      es_manager.train.env_groups=1
      es_manager.train.group_size=1
      es_manager.train.env_configs.tags=[${DATASET_TAG}]
      es_manager.train.env_configs.n_groups=[1]
      es_manager.val.env_groups=${batch_size}
      es_manager.val.group_size=${EVAL_GROUP_SIZE}
      es_manager.val.env_configs.tags=[${DATASET_TAG}]
      es_manager.val.env_configs.n_groups=[${batch_size}]
      custom_envs.MetamathQA.max_actions_per_traj=${MAX_ACTIONS_PER_TRAJ}
      agent_proxy.max_turn=${EVAL_MAX_TURN}
      val_agent_proxy.max_turn=${EVAL_MAX_TURN}
      actor_rollout_ref.rollout.val_kwargs.do_sample=false
      actor_rollout_ref.rollout.gpu_memory_utilization=${EVAL_GPU_MEMORY_UTIL}
      actor_rollout_ref.rollout.response_length=${EVAL_RESPONSE_LENGTH}
      actor_rollout_ref.rollout.max_model_len=${EVAL_MAX_MODEL_LEN}
      actor_rollout_ref.rollout.load_format=${ROLLOUT_LOAD_FORMAT}
      actor_rollout_ref.rollout.layered_summon=${ROLLOUT_LAYERED_SUMMON_BOOL}
      +output.dir="${batch_out_dir}"
      +output.filename="rollouts.pkl"
      +output.append_timestamp=false
      trainer.logger=[console]
    )

    WANDB_PROJECT="${WANDB_PROJECT}" WANDB_NAME="${eval_key}_batch${batch}" WANDB_RUN_ID="${eval_key}_batch${batch}" \
    ${PYTHON_BIN} -m ufb.llm_agent.agent_proxy "${EVAL_CMD_OVERRIDES[@]}"
  done

  # 合并所有批次的 turn_details.json
  echo "[EVAL] 合并所有批次的结果..."
  ${PYTHON_BIN} -c "
import json
import os
from pathlib import Path

out_dir = Path('${out_dir}')
all_details = []

for batch in range(${EVAL_NUM_BATCHES}):
    batch_file = out_dir / f'batch_{batch}' / 'turn_details.json'
    if batch_file.exists():
        with open(batch_file) as f:
            all_details.extend(json.load(f))

# 保存合并后的结果
with open(out_dir / 'turn_details.json', 'w') as f:
    json.dump(all_details, f, indent=2, ensure_ascii=False)

print(f'[INFO] 合并了 {len(all_details)} 个问题的结果')
"
}

run_turn_analysis() {
  echo "[ANALYSIS] 汇总分轮准确率..."

  local input_dirs=()
  local model_names=()

  for idx in "${!eval_labels[@]}"; do
    local label="${eval_labels[$idx]}"
    local out_dir="${RUN_OUT_DIR}/${label}"
    if [[ -f "${out_dir}/turn_details.json" ]]; then
      input_dirs+=("${out_dir}")
      model_names+=("${label}")
    else
      echo "[WARN] 跳过缺失 turn_details.json: ${out_dir}"
    fi
  done

  if [[ ${#input_dirs[@]} -eq 0 ]]; then
    echo "[WARN] 没有找到任何 turn_details.json，跳过分轮分析"
    return
  fi

  local analysis_output="${RUN_OUT_DIR}/turn_accuracy_summary.json"

  ${PYTHON_BIN} "${SCRIPT_DIR}/analyze_turns.py" \
    --input_dirs "${input_dirs[@]}" \
    --model_names "${model_names[@]}" \
    --output_file "${analysis_output}" \
    --k_values "${EVAL_K_VALUES[@]}"

  echo "[ANALYSIS] 分轮准确率汇总已保存至 ${analysis_output}"

  # 绘制曲线图
  local plot_output="${RUN_OUT_DIR}/turn_accuracy_plot.png"
  ${PYTHON_BIN} "${SCRIPT_DIR}/plot_turn_accuracy.py" \
    --summary_file "${analysis_output}" \
    --output_file "${plot_output}" \
    --k_values "${EVAL_K_VALUES[@]}"
}

main() {
  declare -a eval_labels=()
  declare -a eval_paths=()

  # 可选：跳过 base_model
  if [[ "${SKIP_BASE_MODEL:-0}" != "1" ]]; then
    eval_labels+=("base_model")
    eval_paths+=("${MODEL_NAME}")
  fi

  for step in "${CKPT_STEPS[@]}"; do
    eval_labels+=("ckpt_step_${step}")
    # Check if HF format exists in actor/huggingface subdirectory
    if [[ -d "${CKPT_DIR}/global_step_${step}/actor/huggingface" ]]; then
      eval_paths+=("${CKPT_DIR}/global_step_${step}/actor/huggingface")
    else
      eval_paths+=("${CKPT_DIR}/global_step_${step}")
    fi
  done

  for idx in "${!eval_labels[@]}"; do
    label="${eval_labels[$idx]}"
    model="${eval_paths[$idx]}"
    if [[ "${label}" != "base_model" && ! -d "${model}" ]]; then
      echo "[WARN] 跳过缺失 ckpt: ${label} -> ${model}"
      continue
    fi
    run_eval "${model}" "${EVAL_RUN_NAME}_${label}_eval_${EVAL_QUESTIONS}" "${RUN_OUT_DIR}/${label}"
  done

  # run_turn_analysis  # Disabled: pass@k is now computed during evaluation

  echo "[DONE] 评估完成，所有输出位于 ${RUN_OUT_DIR}"
}

main "$@"
