# Paper Experiment Hyperparameters

这份文档把当前 paper 叙事、数据映射和仓库里的可运行脚本对齐起来，方便直接整理 appendix 或内部实验表。

对齐来源：

- paper 叙事：`/work/hdd/bfea/cgui/ufb_writing/PAPERNEW.md`
- 实验规划：`/work/hdd/bfea/cgui/ufb_writing/PLAN.md`
- 数据落盘映射：`/work/hdd/bfea/cgui/ufb_writing/raw_data/DATA_MAPPING.md`
- 训练脚本：`scripts/experiment_table/_train_standard_experiment.sh`、`scripts/webshop/train_qwen25_3b.sh`、`scripts/exp1_train.sh`
- 评测脚本：`scripts/eval.sh`、`scripts/eval2.sh`、`scripts/eval3.sh`、`scripts/exp1_eval.sh`、`scripts/exp1_eval_own_env.sh`
- 共享配置：`configs/base.yaml`、`configs/envs.yaml`、`configs/ppo_trainer.yaml`

本文档优先记录“脚本真实生效的参数”，而不是 paper 中的简写。

## 1. 先说明几个语义点

| Paper 简写 | 脚本里的真实含义 |
|---|---|
| `T=5`（MetaMathQA） | MetaMathQA 是原生单步环境，真正生效的是 `agent_proxy.max_turn=5`，并且环境内部 `max_steps=5`。`experiment_table` 里写的 `ATTEMPT_TIMES=5`、`TURN_PER_ATTEMPT=5` 对 MetaMathQA 只是记账字段，因为 helper 明确关闭了 retry wrapper。 |
| `one-bit feedback` | 默认不是固定的 `"Incorrect"`，而是从 8 条语义等价 feedback pool 中随机采样。只有在 `randomize_feedback=false` 时才会走固定字符串。 |
| `no feedback` | 对 MetaMathQA 原生 no-feedback 变体，错误后 observation 是空字符串 `""`。对 wrapper 型环境，no-feedback 等价于 `fixed_feedback=""`。 |
| 多轮 retry 环境 | HotpotQA / Sokoban / FrozenLake / WebShop 这类实验，真正的“attempt”由 `MultiTurnRetryWrapper` 决定：`max_retry_attempts` 是 attempt 数，`max_turns_per_attempt` 是每次 attempt 的 turn budget，`agent_proxy.max_turn` 是跨所有 attempt 的总 turn budget。Countdown 不是 wrapper 环境，而是原生单步环境，attempt 直接等于 turn。 |
| paper 里的 reward | MetaMathQA 标准设置是 `r_t = 0.5^(t-1)`，错误为 `0`，episode 末尾再减 repetition penalty。wrapper 型环境则是在每个 attempt 成功或失败后，再乘上 `1 / reward_decay_base^attempt_num`。 |

## 2. 训练脚本家族

### 2.1 `scripts/experiment_table/_train_standard_experiment.sh`

这一家族覆盖：

- MetaMathQA one-bit / no-feedback / specific / success-first
- HotpotQA one-bit
- MetaMathQA 的 scale / cross-family 训练

共享训练超参数如下。

| 项目 | 数值 |
|---|---|
| Optimizer | PPO |
| 总训练步数 | `200` |
| 存 checkpoint | 每 `50` step |
| 验证频率 | 每 `10` step |
| 训练 batch 组织 | `8` env groups × `16` rollouts = `128` rollouts / step |
| 训练筛选 | `rollout_filter_ratio=0.25`，`rollout_filter_type=std` |
| 验证 batch 组织 | `512` env groups × `1` |
| Actor learning rate | `1e-6` |
| Actor betas | `[0.9, 0.999]` |
| Critic learning rate | `1e-5` |
| Critic betas | `[0.9, 0.999]` |
| PPO mini-batch size | `32` |
| PPO micro-batch / GPU | `2` |
| Log-prob micro-batch / GPU | `4` |
| PPO clip ratio | `[0.2, 0.28]` |
| Entropy coefficient | `0.001` |
| KL 设置 | `use_kl_loss=false`，`kl_coef=0.001`，`kl_ctrl.type=fixed` |
| Advantage estimator | `gae`，`gamma=1.0`，`lam=1.0` |
| Rollout temperature | `1.0` |
| Validation temperature | `0.5` |
| `top_p` / `top_k` | `1` / `-1` |
| Response length | `400` |
| Context window | `full`，`max_context_window=-1` |
| Tensor parallel size | `1` |
| GPU memory utilization | `0.3` |
| `max_model_len` | `8192` |
| `max_num_batched_tokens` | `16384` |
| Keep actor checkpoints | `4` |
| Keep critic checkpoints | `0` |

这组脚本还有三点需要单独记：

- `MetaMathQA*` tag 会显式关闭 retry wrapper，所以真正生效的是原生 MetaMathQA 环境，而不是 wrapper。
- helper 默认 `NGPUS=1`。因此大多数 `experiment_table/*.sh` 虽然 Slurm 申请了 `2` 张卡，实际训练默认只会把 `CUDA_VISIBLE_DEVICES` 设成 `0`，也就是只用 `1` 张卡。例外是 `metamathqa_qwen25_7b_one_bit.sh`，它显式导出了 `NGPUS=4`。
- `hotpotqa_qwen25_3b_one_bit.sh` 通过 `++custom_envs.HotpotQA.retry.*` 动态给 HotpotQA 加上 retry wrapper，因此它虽然是 QA 环境，训练协议仍然是 `5 attempts × 1 turn/attempt`。

### 2.2 `scripts/webshop/train_qwen25_3b.sh`

这一家族只覆盖 WebShop，且参数和标准 helper 不完全一样。

| 项目 | 数值 |
|---|---|
| Optimizer | PPO |
| 总训练步数 | `200` |
| 存 checkpoint | 每 `50` step |
| 验证频率 | 每 `10` step |
| GPU 数 | 默认 `2` |
| 训练 batch 组织 | `8` env groups × `16` rollouts |
| 验证 batch 组织 | `512` env groups × `1` |
| PPO mini-batch size | `32` |
| PPO micro-batch / GPU | `2` |
| Log-prob micro-batch / GPU | `4` |
| Rollout temperature | `1.0` |
| Validation temperature | `0.5` |
| Response length | `400` |
| `max_model_len` | `8100` |
| `max_num_batched_tokens` | `16384` |
| Tensor parallel size | `1` |
| GPU memory utilization | `0.3` |
| Reward decay base | `2.0` |
| 当前 launcher 的 retry 配置 | `2 attempts × 7 turns/attempt`，`max_turn=14` |
| Action budget | `max_actions_per_turn=1`，`max_actions_per_attempt=7`，`max_actions_per_traj=7` |

这里和 paper 简写差异最大：当前可运行的 WebShop launcher 不是 `5 attempts`，而是 `2 attempts × 7 turns`。

### 2.3 `scripts/exp1_train.sh`

这一家族主要覆盖早期 Exp1/跨域实验里的：

- Countdown
- SimpleSokoban
- FrozenLake
- 也可用于 MetaMathQA，但现在 MetaMathQA/HotpotQA/WebShop 主结果更接近 `experiment_table` 家族

它和上面的标准 helper 有四个关键差异：

| 项目 | 数值 |
|---|---|
| 总训练步数 | `200` |
| 存 checkpoint | 每 `50` step |
| 训练 batch 组织 | `8 × 16` |
| 验证 batch 组织 | `512 × 1` |
| PPO mini-batch size | `8` |
| PPO micro-batch / GPU | `2` |
| Log-prob micro-batch / GPU | `8` |
| Entropy coefficient | `0` |
| Response length | `512` |
| Tensor parallel size | `1` |
| GPU memory utilization | `0.3` |
| `max_model_len` | `8192` |
| `max_num_batched_tokens` | `16384` |
| 默认 GPU 数 | `2` |

环境预算方面：

| 环境 | `exp1_train.sh` 生效设置 |
|---|---|
| MetaMathQA | `max_turn=5` |
| Countdown | `max_turn=5` |
| SimpleSokoban | `max_turn=15`，但 retry wrapper 参数来自 `configs/envs.yaml`，实际是 `3 attempts × 5 turns/attempt × 10 actions/attempt` |
| FrozenLake | `max_turn=15`，同样实际是 `3 attempts × 5 turns/attempt × 10 actions/attempt` |

这意味着 `exp1_train.sh` 下的 Sokoban / FrozenLake 训练并不是 paper 文字里常写的 `5 attempts`，而是当前配置里的 `3 attempts`。

## 3. 评测脚本家族

### 3.1 `scripts/eval.sh`

这是当前 MetaMathQA / HotpotQA / WebShop 结果最通用的评测入口。默认 one-bit feedback。

| 项目 | 数值 |
|---|---|
| Eval only | `trainer.total_training_steps=0` |
| 默认 GPU 数 | `2` |
| 默认验证样本 | `VAL_GROUPS=1024` |
| PPO mini-batch size | `16` |
| PPO micro-batch / GPU | `2` |
| Tensor parallel size | `1` |
| GPU memory utilization | `0.75` |
| Response length | 继承配置，当前实验默认是 `400` |
| Validation temperature | `0.5` |
| `top_p` / `top_k` | `1.0` / `-1` |
| 默认反馈模式 | `one-bit` |

可通过：

- `FEEDBACK_MODE=no-feedback`
- `FEEDBACK_MODE=specific` 并配合 `FIXED_FEEDBACK=...`
- `EVAL_TURN=...`

来覆盖反馈或最大 turn。

### 3.2 `scripts/eval2.sh` 与 `scripts/eval3.sh`

这两个脚本本质上是 `eval.sh` 的 convenience wrapper：

| 脚本 | 默认环境 | 默认 `ENV_TAG` | 默认 `FEEDBACK_MODE` |
|---|---|---|---|
| `eval2.sh` | `hotpotqa` | `HotpotQA` | `no-feedback` |
| `eval3.sh` | `webshop` | `WebShop` | `no-feedback` |

其余超参数与 `eval.sh` 相同。

### 3.3 `scripts/exp1_eval.sh`

这是 Countdown / Sokoban / FrozenLake / MetaMathQA 的跨环境 generalization eval。

| 项目 | 数值 |
|---|---|
| 默认环境集合 | `MetamathQA Countdown SimpleSokoban FrozenLake` |
| 每个环境验证样本 | `1024` |
| 默认最大 attempt | `5` |
| 默认 feedback | `one-bit` |
| 默认 GPU 数 | `1` |
| PPO mini-batch size | `16` |
| PPO micro-batch / GPU | `2` |
| Log-prob micro-batch / GPU | `2` |
| Response length | `400` |
| Tensor parallel size | `1` |
| GPU memory utilization | `0.75` |
| `max_model_len` | `14000` |
| `max_num_batched_tokens` | `14000` |

环境 attempt 语义：

| 环境 | attempt 来源 |
|---|---|
| MetaMathQA / Countdown | turn index |
| SimpleSokoban / FrozenLake | wrapper 的 `attempt_num` |

注意：`exp1_eval.sh` 会把 Sokoban / FrozenLake 的评测 attempt 预算统一改成 `5`，因此它与 `exp1_train.sh` 的 `3 attempts` 训练设置并不完全相同。

### 3.4 `scripts/exp1_eval_own_env.sh`

这个脚本和 `exp1_eval.sh` 基本一致，但每个模型只在自己的训练环境上做评测，主要用于：

- `step 50 / 100 / 150 / 200` own-env 曲线
- Table 1 的 trained own-domain 结果

共享超参数与 `exp1_eval.sh` 相同。

## 4. MetaMathQA 变体的有效配置

| 变体 | 训练 launcher | 环境 tag / env type | feedback | reward |
|---|---|---|---|---|
| Standard multi-turn | `scripts/experiment_table/metamathqa_qwen25_3b_one_bit.sh` | `MetamathQA` / `metamathqa` | 随机 one-bit pool | 正确时 `1, 0.5, 0.25, 0.125, 0.0625`；错误 `0`；episode 末尾 repetition penalty |
| No feedback | 当前 `scripts/experiment_table/` 没有 Qwen-3B 专门 launcher；可用同一 helper 配 `ENV_TAG=MetamathQANoFeedback` 复现 | `MetamathQANoFeedback` / `metamathqa_no_feedback` | 空字符串 | 与 standard 相同，只是错误后不给反馈 |
| Specific feedback | `scripts/experiment_table/metamathqa_qwen25_3b_specific.sh` | `MetamathQASpecificFeedback` / `metamathqa_specific_feedback` | 基于 ground truth 的方向性提示 | 与 standard 相同 |
| No-recovery-reward | `scripts/experiment_table/metamathqa_qwen25_3b_success_first.sh` | `MetamathQAFirstTurnSuccess` / `metamathqa_first_turn_success` | 随机 one-bit pool | 只有首轮答对给 `1`；后续纠正奖励为 `0`；episode 末尾仍保留 repetition penalty |

## 5. Paper 结果和脚本的对应关系

下表按当前 paper 叙事组织。

| Paper 结果 | 训练模型 / 条件 | 训练脚本 | 评测脚本 | 备注 |
|---|---|---|---|---|
| Fig 1: base vs single-turn vs multi-turn on MetaMathQA | multi-turn RL: Qwen2.5-3B + one-bit；single-turn RL: 手工数据文件；base: 无训练 | multi-turn 用 `scripts/experiment_table/metamathqa_qwen25_3b_one_bit.sh`；single-turn 的 launcher 当前不在 `scripts/` 中 | base / multi-turn 可用 `scripts/eval.sh`；single-turn 来自 `ufb_writing/table_figures/.../singleturn_eval.json` | 当前仓库里没有 single-turn RL 的可运行脚本 |
| Fig 2: base self-correction with vs without feedback | 无训练 | 无 | `scripts/eval.sh`，分别跑 `FEEDBACK_MODE=one-bit` 和 `FEEDBACK_MODE=no-feedback` | 对应 `PAPERNEW.md` §3.1 的 base self-correction |
| Fig 3: reward ablation | standard multi-turn / success-first / single-turn | standard: `metamathqa_qwen25_3b_one_bit.sh`；success-first: `metamathqa_qwen25_3b_success_first.sh`；single-turn 仍是手工数据 | `scripts/eval.sh` | success-first 是当前脚本里最直接的 no-recovery-reward 实现 |
| Table 1: cross-domain consistency | MetaMathQA / HotpotQA / WebShop / Countdown / Sokoban / FrozenLake | MetaMathQA, HotpotQA 用 `experiment_table/*`；WebShop 用 `scripts/webshop/train_qwen25_3b.sh`；Countdown/Sokoban/FrozenLake 用 `scripts/exp1_train.sh` | MetaMathQA/HotpotQA/WebShop 用 `eval.sh`/`eval2.sh`/`eval3.sh`；Countdown/Sokoban/FrozenLake 用 `exp1_eval.sh` 或 `exp1_eval_own_env.sh` | 这张表混用了两套训练脚本家族 |
| Fig 4: cross-domain transfer matrix | 6 个 domain 的 trained model 交叉评测 | 训练来源同 Table 1 | 主要用 `scripts/exp1_eval.sh` 风格的跨环境 eval；HotpotQA / WebShop 结果在 `DATA_MAPPING.md` 里来自单独 eval 数据集 | transfer 是评测侧统一得更强，训练侧仍是混合来源 |
| Fig 5: information scaling | no feedback / one-bit / generic / specific | one-bit: `metamathqa_qwen25_3b_one_bit.sh`；specific: `metamathqa_qwen25_3b_specific.sh`；no-feedback 训练 launcher 当前未在 `experiment_table` 中给出 | `scripts/eval.sh` 跑不同 `FEEDBACK_MODE` 或 env tag | generic hint 的训练 launcher 也不在当前 `scripts/experiment_table/` 中 |
| Appendix D: signal validity | base model only | 无 | `scripts/eval.sh`，分别用 one-bit / `FIXED_FEEDBACK=\"Please continue\"` / no-feedback | 这是 inference-only 实验 |
| Appendix F: scale / cross-family | Qwen 1.5B / 3B / 7B；Llama / Gemma / Phi | 对应 `scripts/experiment_table/metamathqa_*_{one_bit,no_feedback}.sh` | `scripts/eval.sh` | `Qwen2.5-7B` 脚本显式用 `4` GPUs，`GPU_MEMORY_UTILIZATION=0.5` |

## 6. 当前仓库里无法直接从 `scripts/` 恢复的实验

以下结果在 `DATA_MAPPING.md` 里存在，但当前 `scripts/` 下没有直接可跑的对应 launcher，整理 paper 表格时建议单独标注“manual / external”来源：

- Single-turn RL 的训练与评测结果
- MetaMathQA Qwen2.5-3B 的 no-feedback 专门训练 launcher
- generic hint 训练 launcher

如果只是写 appendix 超参数表，建议把这些行写成：

- “same PPO hyperparameters as standard MetaMathQA multi-turn run”
- 再单独在备注里标 `result loaded from manual JSON/CSV; launcher not present in current repo`

## 7. Appendix 可直接抄的最小表述

如果只想在 paper appendix 里放一张紧凑表，可以直接用下面这段作为骨架：

> Unless otherwise noted, all RL runs use PPO for 200 steps with actor learning rate `1e-6`, critic learning rate `1e-5`, batch size `8 × 16 = 128` rollouts per update, top-25% group filtering by reward standard deviation, asymmetric clip ratio `[0.2, 0.28]`, KL coefficient `0.001`, and maximum interaction horizon `T=5`. For MetaMathQA, one-bit feedback is sampled from an 8-template pool, rewards follow `r_t = 0.5^(t-1)` for the first correct answer at turn `t`, and a repetition penalty is applied at episode end. Most MetaMathQA / HotpotQA runs use `ppo_mini_batch_size=32`, `ppo_micro_batch_size_per_gpu=2`, `log_prob_micro_batch_size_per_gpu=4`, and `response_length=400`. The legacy Exp1 runs for Countdown / Sokoban / FrozenLake instead use `ppo_mini_batch_size=8`, `log_prob_micro_batch_size_per_gpu=8`, `entropy_coeff=0`, and `response_length=512`. Evaluation uses the same interaction protocol as training, typically with `1024` validation instances per condition and sampling temperature `0.5`.
