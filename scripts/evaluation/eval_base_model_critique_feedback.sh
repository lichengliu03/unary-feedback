#!/usr/bin/env bash

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
MODEL_NAME="${MODEL_NAME:-qwen25_3b_base_critique}"
WANDB_PROJECT="${WANDB_PROJECT:-ragen_metamathqa_critique}"
EVAL_MAX_TURN="${EVAL_MAX_TURN:-5}"
EVAL_GPU_MEMORY_UTIL="${EVAL_GPU_MEMORY_UTIL:-0.4}"
EVAL_RESPONSE_LENGTH="${EVAL_RESPONSE_LENGTH:-256}"
EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-16384}"
ROLLOUT_LOAD_FORMAT="${ROLLOUT_LOAD_FORMAT:-auto}"
ROLLOUT_LAYERED_SUMMON="${ROLLOUT_LAYERED_SUMMON:-1}"
DATASET_TAG="MetamathQACritique"  # 使用 Critique 环境
MAX_ACTIONS_PER_TRAJ="${MAX_ACTIONS_PER_TRAJ:-5}"

EVAL_QUESTIONS="${EVAL_QUESTIONS:-1024}"
EVAL_GROUP_SIZE="${EVAL_GROUP_SIZE:-1}"
EVAL_K_VALUES_ENV="${EVAL_K_VALUES:-1 2 3 4 5}"
read -r -a EVAL_K_VALUES <<< "${EVAL_K_VALUES_ENV}"

# ====== 批处理配置 ======
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
if (( EVAL_QUESTIONS % EVAL_BATCH_SIZE != 0 )); then
  echo "[WARN] EVAL_QUESTIONS (${EVAL_QUESTIONS}) 不能被 EVAL_BATCH_SIZE (${EVAL_BATCH_SIZE}) 整除，将向上取整"
fi
EVAL_NUM_BATCHES=$(( (EVAL_QUESTIONS + EVAL_BATCH_SIZE - 1) / EVAL_BATCH_SIZE ))

echo "[INFO] 将 ${EVAL_QUESTIONS} 道题分成 ${EVAL_NUM_BATCHES} 批，每批 ${EVAL_BATCH_SIZE} 道题"

# ====== 输出目录 ======
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EVAL_RUN_NAME="${EVAL_RUN_NAME:-eval_base_critique_${TIMESTAMP}}"
OUT_BASE="${RESULT_BASE_DIR:-${REPO_DIR}/result}/critique_experiments"
mkdir -p "${OUT_BASE}"
RUN_OUT_DIR="${OUT_BASE}/${EVAL_RUN_NAME}"
mkdir -p "${RUN_OUT_DIR}"

echo "[INFO] Evaluating base model with CRITIQUE environment: ${MODEL_PATH}"
echo "[INFO] Model name: ${MODEL_NAME}"
echo "[INFO] Output dir: ${RUN_OUT_DIR}"
echo "[INFO] Eval questions: ${EVAL_QUESTIONS}, max_turn: ${EVAL_MAX_TURN}"
echo "[INFO] Using MetamathQACritique environment"

ROLLOUT_LAYERED_SUMMON_BOOL="False"
if [[ "${ROLLOUT_LAYERED_SUMMON}" == "1" || "${ROLLOUT_LAYERED_SUMMON}" == "true" ]]; then
  ROLLOUT_LAYERED_SUMMON_BOOL="True"
fi

# ====== 运行评估 ======
eval_key="${EVAL_RUN_NAME}_${MODEL_NAME}_eval_${EVAL_QUESTIONS}"
out_dir="${RUN_OUT_DIR}"

echo "[EVAL] ${eval_key} -> ${MODEL_PATH}"

# 分批运行评估
for (( batch=0; batch<EVAL_NUM_BATCHES; batch++ )); do
  batch_start=$(( batch * EVAL_BATCH_SIZE ))
  batch_end=$(( (batch + 1) * EVAL_BATCH_SIZE ))
  if (( batch_end > EVAL_QUESTIONS )); then
    batch_end=${EVAL_QUESTIONS}
  fi
  batch_size=$(( batch_end - batch_start ))

  echo "[EVAL] 批次 $((batch+1))/${EVAL_NUM_BATCHES}: 题目 ${batch_start}-${batch_end} (共 ${batch_size} 题)"

  batch_out_dir="${out_dir}/batch_${batch}"
  mkdir -p "${batch_out_dir}"

  EVAL_CMD_OVERRIDES=(
    --config-name train_critique
    trainer.experiment_name="${eval_key}_batch${batch}"
    trainer.project_name="${WANDB_PROJECT}"
    actor_rollout_ref.model.path="${MODEL_PATH}"
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
    custom_envs.MetamathQACritique.max_actions_per_traj=${MAX_ACTIONS_PER_TRAJ}
    agent_proxy.max_turn=${EVAL_MAX_TURN}
    val_agent_proxy.max_turn=${EVAL_MAX_TURN}
    actor_rollout_ref.rollout.val_kwargs.do_sample=false
    actor_rollout_ref.rollout.gpu_memory_utilization=${EVAL_GPU_MEMORY_UTIL}
    actor_rollout_ref.rollout.response_length=${EVAL_RESPONSE_LENGTH}
    actor_rollout_ref.rollout.max_model_len=${EVAL_MAX_MODEL_LEN}
    actor_rollout_ref.rollout.load_format=${ROLLOUT_LOAD_FORMAT}
    +actor_rollout_ref.rollout.layered_summon=${ROLLOUT_LAYERED_SUMMON_BOOL}
    +output.dir="${batch_out_dir}"
    +output.filename="rollouts.pkl"
    +output.append_timestamp=false
    trainer.logger=[console]
  )

  WANDB_PROJECT="${WANDB_PROJECT}" WANDB_NAME="${eval_key}_batch${batch}" WANDB_RUN_ID="${eval_key}_batch${batch}" \
  ${PYTHON_BIN} -m ufb.eval "${EVAL_CMD_OVERRIDES[@]}"
done

# 合并所有批次的 turn_details.json
echo "[EVAL] 合并所有批次的结果..."
${PYTHON_BIN} -c "
import json
import os
from pathlib import Path

out_dir = Path('${out_dir}')
all_details = []

for batch_dir in sorted(out_dir.glob('batch_*')):
    turn_details_file = batch_dir / 'turn_details.json'
    if turn_details_file.exists() and turn_details_file.stat().st_size > 0:
        with open(turn_details_file, 'r') as f:
            try:
                batch_data = json.load(f)
                if isinstance(batch_data, list):
                    all_details.extend(batch_data)
            except json.JSONDecodeError:
                print(f'[WARN] 无法解析 {turn_details_file}')

# 保存合并后的结果
output_file = out_dir / 'turn_details.json'
with open(output_file, 'w') as f:
    json.dump(all_details, f, indent=2)

print(f'[INFO] 合并了 {len(all_details)} 个问题的结果')
print(f'[INFO] 结果保存在: {output_file}')
"

echo "[DONE] 评估完成，所有输出位于 ${RUN_OUT_DIR}"
echo ""
echo "查看结果:"
echo "  turn_details.json: ${RUN_OUT_DIR}/turn_details.json"
echo ""
echo "计算 pass@k 指标:"
echo "  python tools/compute_passk.py ${RUN_OUT_DIR}/turn_details.json"
